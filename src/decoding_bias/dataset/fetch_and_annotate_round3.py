"""
Round-3 family-centric expansion orchestrator.

Reads dataset_update/expansion_round3_family_targets.csv (one row per
(protein_family, target_domain, move) cell with a pre-built UniProt query)
and produces dataset_update/round3/expansion_round3_for_scoring.csv ready
for the scoring pipeline.

Stages (same shape as fetch_and_annotate_round2.py, with family-centric
finalize logic):
  queries     -- pass-through; copies queries from the target CSV
  fetch       -- query UniProt per cell, Swiss-Prot only
  filter      -- dedup against the expanded exclusion pool
                 (main + merged_dataset + round-2 KEPT + round-2 DROPPED)
  structures  -- download AF models
  features    -- sequence + structural features
  classify    -- protein_family, broad_function, flags
  finalize    -- per-cell quotas, per-species cap → CSV

Run:
    python fetch_and_annotate_round3.py --stage all
    python fetch_and_annotate_round3.py --stage queries --dry-run
    python fetch_and_annotate_round3.py --stage fetch --limit 5  # smoke test
"""

import argparse
import json
import re
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from decoding_bias.features.sequence_features import calculate_sequence_features
from decoding_bias.features.structural_features import download_alphafold_structure, extract_features
from dataset_update.protein_classification import classify

ROUND3 = HERE / "round3"
ROUND3.mkdir(exist_ok=True)
PDB_CACHE = HERE / "alphafold_cache"
PDB_CACHE.mkdir(exist_ok=True)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
TARGETS_CSV = HERE / "expansion_round3_family_targets.csv"

OVERSAMPLE_FACTOR = 3
PER_CELL_MAX_FETCH = 200
PER_SPECIES_CAP = 3        # within a (family, domain) cell, no more than 3 from one species
MIN_PLDDT = 70.0
MIN_LEN = 50
MAX_LEN = 1000

FIELDS = ",".join([
    "accession", "id", "protein_name", "gene_names",
    "organism_name", "organism_id", "lineage",
    "length", "sequence", "ec", "keyword",
    "cc_function", "cc_subcellular_location", "cc_subunit",
    "xref_pfam", "protein_families",
])


def _norm_species(name):
    if pd.isna(name):
        return name
    name = re.sub(r"\s*\(strain[^)]*\)", "", str(name))
    name = re.sub(r"\s*\([^)]+\)", "", name).strip()
    return name


# ---- Stage 1: queries (pass-through from the target CSV) ----

def stage_queries(dry_run=False):
    gap = pd.read_csv(TARGETS_CSV)
    queries = []
    for _, r in gap.iterrows():
        target_n = int(r["target_n"])
        if target_n <= 0:
            continue
        fetch_n = min(target_n * OVERSAMPLE_FACTOR, PER_CELL_MAX_FETCH)
        cell_id = (f"{r['protein_family']}__{r['move']}__"
                   f"{r['target_domain']}")
        queries.append({
            "cell_id": cell_id,
            "protein_family": r["protein_family"],
            "target_domain": r["target_domain"],
            "move": r["move"],
            "target_n": target_n,
            "fetch_n": fetch_n,
            "query": r["query"],
        })
    (ROUND3 / "queries.json").write_text(json.dumps(queries, indent=2))
    print(f"Built {len(queries)} queries; target={sum(q['target_n'] for q in queries)}; "
          f"fetch={sum(q['fetch_n'] for q in queries)}")
    if dry_run:
        print("\nSample queries:")
        for q in queries[:5]:
            print(f"  [{q['move']}] {q['protein_family'][:60]} → {q['target_domain']} "
                  f"(target={q['target_n']})")
            print(f"    {q['query']}")


def _uniprot_paged_search(query, want_n, fields=FIELDS):
    rows, fetched, cursor = [], 0, None
    while fetched < want_n:
        params = {"query": query, "format": "tsv", "fields": fields,
                  "size": min(500, want_n - fetched)}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(UNIPROT_SEARCH, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  network error: {e}", flush=True)
            break
        if r.status_code != 200:
            print(f"  UniProt {r.status_code}: {r.text[:200]}", flush=True)
            break
        page = pd.read_csv(StringIO(r.text), sep="\t")
        if len(page) == 0:
            break
        rows.append(page)
        fetched += len(page)
        link = r.headers.get("Link", "")
        cursor = None
        if 'rel="next"' in link:
            m = re.search(r"cursor=([^&>]+)", link)
            if m:
                cursor = m.group(1)
        if cursor is None:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).head(want_n)


def stage_fetch(dry_run=False, limit=None):
    queries = json.loads((ROUND3 / "queries.json").read_text())
    if limit:
        queries = queries[:limit]
    if dry_run:
        print(f"Would fetch {sum(q['fetch_n'] for q in queries)} entries "
              f"across {len(queries)} cells")
        return
    all_rows, yield_log = [], []
    for q in tqdm(queries, desc="UniProt cells"):
        page = _uniprot_paged_search(q["query"], q["fetch_n"])
        yield_log.append({"cell_id": q["cell_id"], "target_n": q["target_n"],
                          "fetch_n": q["fetch_n"], "got": len(page)})
        if len(page):
            page["cell_id"] = q["cell_id"]
            page["target_protein_family"] = q["protein_family"]
            page["target_domain"] = q["target_domain"]
            page["target_n"] = q["target_n"]
            page["move"] = q["move"]
            all_rows.append(page)
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(ROUND3 / "candidates_raw.csv", index=False)
        pd.DataFrame(yield_log).to_csv(ROUND3 / "fetch_yield_log.csv", index=False)
        print(f"Fetched {len(out)} candidate rows (with overlap); "
              f"unique Entries: {out['Entry'].nunique() if 'Entry' in out.columns else 0}")
    else:
        print("No results returned from UniProt.")


# ---- Stage 3: filter ----

def stage_filter():
    raw = pd.read_csv(ROUND3 / "candidates_raw.csv")
    rename = {
        "Entry": "Entry", "Entry Name": "EntryName",
        "Protein names": "protein_name", "Gene Names": "gene_names",
        "Organism": "Organism", "Organism (ID)": "organism_id",
        "Lineage": "lineage", "Taxonomic lineage": "lineage",
        "Length": "Length", "Sequence": "sequence",
        "EC number": "ec", "Keywords": "keywords",
        "Function [CC]": "cc_function",
        "Subcellular location [CC]": "cc_subcellular_location",
        "Subunit structure [CC]": "cc_subunit",
        "Pfam": "Pfam", "Protein families": "protein_families_raw",
    }
    raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    # Build exclusion pool: every Entry we have ever seen
    pool = set()
    for path, label in [
        (HERE / "Decoding_Bias_Dataset_updated.csv", "main"),
        (HERE / "merged_dataset.csv", "merged (main+round1)"),
        (HERE / "round2" / "expansion_round2_KEPT.csv", "round2 KEPT"),
        (HERE / "round2" / "expansion_round2_DROPPED.csv", "round2 DROPPED"),
    ]:
        if path.exists():
            ents = set(pd.read_csv(path, usecols=["Entry"])["Entry"])
            pool |= ents
            print(f"  Exclusion pool from {label}: +{len(ents)} (total {len(pool)})")
    print(f"Final exclusion pool: {len(pool)} unique Entries")

    before = len(raw)
    raw = raw.drop_duplicates(subset=["Entry"], keep="first")
    print(f"After dedup within fetch: {len(raw)} (was {before})")

    raw = raw[(raw["Length"] >= MIN_LEN) & (raw["Length"] <= MAX_LEN)]
    print(f"After length filter: {len(raw)}")

    raw = raw[~raw["Entry"].isin(pool)]
    print(f"After exclusion-pool filter: {len(raw)}")

    def _ok_seq(s):
        if not isinstance(s, str) or not s:
            return False
        bad = set(s.upper()) - set("ACDEFGHIKLMNPQRSTVWY")
        return len(bad) == 0 or bad <= {"U", "X"}

    raw = raw[raw["sequence"].apply(_ok_seq)]
    print(f"After sequence-validity filter: {len(raw)}")

    raw["species"] = raw["Organism"].apply(_norm_species)

    def _domain(line):
        if not isinstance(line, str):
            return None
        for d in ("Viruses", "Archaea", "Bacteria", "Eukaryota"):
            if d in line:
                return d
        return None
    raw["domain"] = raw["lineage"].apply(_domain) if "lineage" in raw.columns else None
    # If lineage missing, backfill domain from target_domain
    if "target_domain" in raw.columns:
        raw["domain"] = raw["domain"].fillna(raw["target_domain"])

    raw.to_csv(ROUND3 / "candidates_filtered.csv", index=False)
    print(f"Wrote {ROUND3/'candidates_filtered.csv'}: {len(raw)} candidates")


# ---- Stage 4: structures ----

def stage_structures(limit=None):
    df = pd.read_csv(ROUND3 / "candidates_filtered.csv")
    if limit:
        df = df.head(limit)
    status = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="AF download"):
        entry = row["Entry"]
        # Be permissive about the model version
        existing = list(PDB_CACHE.glob(f"AF-{entry}-F1-model_v*.pdb"))
        if existing:
            status.append({"Entry": entry, "ok": True, "path": str(existing[0]), "cached": True})
            continue
        try:
            result = download_alphafold_structure(entry, str(PDB_CACHE))
            ok = result is not None and Path(result).exists()
            status.append({"Entry": entry, "ok": ok,
                           "path": str(result) if ok else "", "cached": False})
        except Exception as e:
            status.append({"Entry": entry, "ok": False, "path": "",
                           "error": str(e), "cached": False})
        time.sleep(0.03)
    s = pd.DataFrame(status)
    s.to_csv(ROUND3 / "structure_status.csv", index=False)
    print(f"AF success: {s['ok'].sum()}/{len(s)}")


# ---- Stage 5: features ----

def stage_features():
    df = pd.read_csv(ROUND3 / "candidates_filtered.csv")
    status = pd.read_csv(ROUND3 / "structure_status.csv")
    have = set(status.loc[status["ok"], "Entry"])
    df = df[df["Entry"].isin(have)].copy()
    print(f"Computing features for {len(df)} entries")

    seq_feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="seq features"):
        f = calculate_sequence_features(str(row["sequence"]))
        f["Entry"] = row["Entry"]
        seq_feats.append(f)
    seq_df = pd.DataFrame(seq_feats)

    struct_feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="struct features"):
        existing = list(PDB_CACHE.glob(f"AF-{row['Entry']}-F1-model_v*.pdb"))
        path = existing[0] if existing else None
        if path is None:
            struct_feats.append({"Entry": row["Entry"], "_err": "no_structure"})
            continue
        try:
            f = extract_features(row["Entry"], str(path))
            struct_feats.append(f)
        except Exception as e:
            struct_feats.append({"Entry": row["Entry"], "_err": str(e)})
    struct_df = pd.DataFrame(struct_feats)

    out = df.merge(seq_df, on="Entry", how="left", suffixes=("", "_seq"))
    out = out.merge(struct_df, on="Entry", how="left", suffixes=("", "_struct"))

    if "avg_plddt" in out.columns:
        before = len(out)
        out = out[out["avg_plddt"].fillna(0) >= MIN_PLDDT]
        print(f"pLDDT≥{MIN_PLDDT} filter: kept {len(out)}/{before}")

    out.to_csv(ROUND3 / "features.csv", index=False)
    print(f"Wrote {ROUND3/'features.csv'}: {len(out)} entries")


# ---- Stage 6: classify ----

def stage_classify():
    df = pd.read_csv(ROUND3 / "features.csv")
    rename = {"Pfam": "Pfam", "ec": "EC number",
              "protein_families_raw": "Protein families",
              "Organism": "Organism", "protein_name": "Protein names"}
    work = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    classified = classify(work)
    out = df.copy()
    for col in ("protein_family", "broad_function", "protein_name_clean",
                "is_enzyme", "is_transmembrane", "is_glycosylated", "has_disordered"):
        if col in classified.columns:
            out[col] = classified[col].values
    out.to_csv(ROUND3 / "classified.csv", index=False)
    print(f"Classified {len(out)} entries; wrote {ROUND3/'classified.csv'}")


# ---- Stage 7: finalize ----

def stage_finalize():
    df = pd.read_csv(ROUND3 / "classified.csv")
    targets = pd.read_csv(TARGETS_CSV)

    # Drop ribosomal/translation_factor stragglers (defensive - query already excluded KW-0689)
    before = len(df)
    df = df[~df["broad_function"].isin({"ribosomal", "translation_factor"})]
    print(f"Defensive drop of ribosomal/translation_factor: {before - len(df)}")

    # Cross-check: did the classifier place the protein in the intended target family?
    # We retain target_protein_family as the cell label for accounting; the classifier
    # may rename to the parent superfamily or vice versa, but the family: UniProt query
    # already restricted by the curated family field, so target_protein_family is
    # the authoritative cell label.
    keep_rows = []
    cell_report = []
    for _, t in targets.iterrows():
        cell_id = f"{t['protein_family']}__{t['move']}__{t['target_domain']}"
        sub = df[df["cell_id"] == cell_id].copy()
        if sub.empty:
            cell_report.append({**t.to_dict(), "kept": 0, "status": "empty"})
            continue
        # Per-species cap inside the cell
        sub = sub.groupby("species", group_keys=False).head(PER_SPECIES_CAP)
        sub = sub.head(int(t["target_n"]))
        cell_report.append({**t.to_dict(), "kept": len(sub), "status": "ok"})
        keep_rows.append(sub)
    if not keep_rows:
        print("Nothing to write.")
        return
    final = pd.concat(keep_rows, ignore_index=True).drop_duplicates("Entry")
    final["source"] = "expansion_round3"
    final["structure_source"] = "AF"
    if "target_domain" in final.columns:
        final["domain"] = final["domain"].fillna(final["target_domain"])

    out = ROUND3 / "expansion_round3_for_scoring.csv"
    final.to_csv(out, index=False)

    report = pd.DataFrame(cell_report)
    report.to_csv(ROUND3 / "finalize_report.csv", index=False)

    print(f"\nFinal round-3 expansion: {len(final)} proteins")
    print(f"  by move: {final['move'].value_counts().to_dict()}")
    print(f"  by domain: {final['domain'].value_counts().to_dict()}")
    print(f"  unique target families: {final['target_protein_family'].nunique()}")
    print(f"  unique species: {final['species'].nunique()}")
    print(f"  ribosomal proteins (should be ~0): "
          f"{(final['broad_function']=='ribosomal').sum()}")
    print(f"Wrote {out}")
    print(f"Wrote {ROUND3/'finalize_report.csv'}")


STAGES = {
    "queries":    stage_queries,
    "fetch":      stage_fetch,
    "filter":     stage_filter,
    "structures": stage_structures,
    "features":   stage_features,
    "classify":   stage_classify,
    "finalize":   stage_finalize,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "all":
        for name in ["queries", "fetch", "filter", "structures",
                     "features", "classify", "finalize"]:
            print(f"\n===== Stage: {name} =====")
            fn = STAGES[name]
            kwargs = {}
            if name in ("queries", "fetch"):
                kwargs["dry_run"] = args.dry_run
            if name in ("fetch", "structures") and args.limit is not None:
                kwargs["limit"] = args.limit
            fn(**kwargs)
    else:
        fn = STAGES[args.stage]
        kwargs = {}
        if args.stage in ("queries", "fetch"):
            kwargs["dry_run"] = args.dry_run
        if args.stage in ("fetch", "structures") and args.limit is not None:
            kwargs["limit"] = args.limit
        fn(**kwargs)


if __name__ == "__main__":
    main()
