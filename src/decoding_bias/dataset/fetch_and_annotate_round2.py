"""
Round-2 expansion orchestrator.

End-to-end: takes the gap-target CSV, queries UniProt (Swiss-Prot only),
downloads AlphaFold structures, computes biophysical + structural features,
runs the existing classifier, and writes a CSV ready for the scoring notebooks.

Stages can be run independently via --stage. All outputs land in
dataset_update/round2/.

Inputs:
  dataset_update/merged_dataset.csv               -- already-included entries (exclude)
  dataset_update/expansion_round2_gap_target.csv  -- per (domain, broad_function) deficits

Outputs (all under dataset_update/round2/):
  queries.json                  -- one UniProt query per cell
  candidates_raw.csv            -- raw query results (Swiss-Prot entries)
  candidates_filtered.csv       -- after duplicate + quality filters
  structure_status.csv          -- AF download success/fail per entry
  features.csv                  -- sequence + structural features
  classified.csv                -- + protein_family + broad_function
  expansion_round2_for_scoring.csv  -- FINAL: ready for scoring pipeline

Usage:
  python fetch_and_annotate_round2.py --stage queries     # build queries.json
  python fetch_and_annotate_round2.py --stage fetch       # query UniProt
  python fetch_and_annotate_round2.py --stage filter      # quality + dedup
  python fetch_and_annotate_round2.py --stage structures  # download AF
  python fetch_and_annotate_round2.py --stage features    # seq + struct features
  python fetch_and_annotate_round2.py --stage classify    # protein_family, broad_function
  python fetch_and_annotate_round2.py --stage finalize    # apply quotas, write final CSV
  python fetch_and_annotate_round2.py --stage all         # run everything
  python fetch_and_annotate_round2.py --dry-run --stage queries  # preview queries
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# --- Make the repo root importable so we can use src.features.* ---
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from decoding_bias.features.sequence_features import calculate_sequence_features
from decoding_bias.features.structural_features import (
    download_alphafold_structure, extract_features, scan_existing_structures,
)
from dataset_update.protein_classification import classify

ROUND2 = HERE / "round2"
ROUND2.mkdir(exist_ok=True)
PDB_CACHE = HERE / "alphafold_cache"
PDB_CACHE.mkdir(exist_ok=True)

# UniProt REST endpoint
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Domain → NCBI taxonomy ID
# Viruses excluded: most viral proteins are not in the AlphaFold DB.
DOMAIN_TAXID = {
    "Archaea": 2157,
    "Bacteria": 2,
    "Eukaryota": 2759,
}

# broad_function → UniProt query fragment
# Use EC for the six enzyme classes; UniProt keywords elsewhere.
FUNCTION_QUERY = {
    "transferase":      'ec:2*',
    "hydrolase":        'ec:3*',
    "oxidoreductase":   'ec:1*',
    "lyase":            'ec:4*',
    "isomerase":        'ec:5*',
    "ligase":           'ec:6*',
    "translocase":      'ec:7*',
    "transcription":    'keyword:KW-0804',
    "membrane":         'keyword:KW-0472',
    "GTPase":           'keyword:KW-0342',
    "electron_carrier": 'keyword:KW-0249',
    "chaperone":        'keyword:KW-0143',
    "signaling":        'keyword:KW-0807',
    "transport":        'keyword:KW-0813',
    "RNA-binding":      'keyword:KW-0694',
    "DNA-binding":      'keyword:KW-0238',
    "cytoskeletal":     'keyword:KW-0206',
    "structural":       'keyword:KW-0729',
    "protease_inhibitor": 'keyword:KW-0646',
}

# Always-applied filters
BASE_FILTER = (
    'reviewed:true AND '
    'length:[50 TO 1000] AND '
    'NOT keyword:KW-0689'  # exclude ribosomal protein
)

# Per-cell: oversample so we have enough survivors after AF + dedup filtering
OVERSAMPLE_FACTOR = 3
PER_CELL_MAX = 1500

# Quality filters applied later
MIN_PLDDT = 70.0
MIN_LEN = 50
MAX_LEN = 1000

# UniProt fields to request (compact set)
FIELDS = ",".join([
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "organism_name",
    "organism_id",
    "lineage",
    "length",
    "sequence",
    "ec",
    "keyword",
    "cc_function",
    "cc_subcellular_location",
    "cc_subunit",
    "xref_pfam",
    "protein_families",
])


# ---------- Stage 1: build queries ----------

def stage_queries(dry_run=False):
    gap = pd.read_csv(HERE / "expansion_round2_gap_target.csv")
    queries = []
    for _, row in gap.iterrows():
        domain, func, add = row["domain"], row["broad_function"], int(row["add_target"])
        if domain not in DOMAIN_TAXID or func not in FUNCTION_QUERY:
            continue
        taxid = DOMAIN_TAXID[domain]
        func_q = FUNCTION_QUERY[func]
        query = f"({BASE_FILTER}) AND (taxonomy_id:{taxid}) AND ({func_q})"
        size = min(add * OVERSAMPLE_FACTOR, PER_CELL_MAX)
        queries.append({
            "cell_id": f"{domain}__{func}",
            "domain": domain,
            "broad_function": func,
            "target_n": add,
            "fetch_n": size,
            "query": query,
        })

    out = ROUND2 / "queries.json"
    (out).write_text(json.dumps(queries, indent=2))
    print(f"Built {len(queries)} queries, total fetch={sum(q['fetch_n'] for q in queries)}, "
          f"target={sum(q['target_n'] for q in queries)}")
    print(f"Wrote {out}")
    if dry_run:
        print("\nFirst 3 queries:")
        for q in queries[:3]:
            print(f"  [{q['cell_id']}] target={q['target_n']} fetch={q['fetch_n']}")
            print(f"    {q['query']}")
    return queries


# ---------- Stage 2: fetch candidates from UniProt ----------

def _uniprot_paged_search(query, want_n, fields=FIELDS):
    """Fetch up to want_n Swiss-Prot entries for a query, with pagination."""
    rows = []
    fetched = 0
    cursor = None
    while fetched < want_n:
        page_size = min(500, want_n - fetched)
        params = {"query": query, "format": "tsv", "fields": fields, "size": page_size}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  UniProt {r.status_code}: {r.text[:200]}")
            break
        # parse TSV
        from io import StringIO
        page = pd.read_csv(StringIO(r.text), sep="\t")
        if len(page) == 0:
            break
        rows.append(page)
        fetched += len(page)
        # cursor for next page
        link = r.headers.get("Link", "")
        cursor = None
        if 'rel="next"' in link:
            import re
            m = re.search(r'cursor=([^&>]+)', link)
            if m:
                cursor = m.group(1)
        if cursor is None:
            break
        time.sleep(0.2)  # polite
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).head(want_n)


def stage_fetch(dry_run=False):
    queries = json.loads((ROUND2 / "queries.json").read_text())
    out_path = ROUND2 / "candidates_raw.csv"
    if dry_run:
        print(f"Would fetch {sum(q['fetch_n'] for q in queries)} entries across "
              f"{len(queries)} cells")
        return
    all_rows = []
    for q in tqdm(queries, desc="UniProt cells"):
        page = _uniprot_paged_search(q["query"], q["fetch_n"])
        if len(page):
            page["cell_id"] = q["cell_id"]
            page["target_domain"] = q["domain"]
            page["target_broad_function"] = q["broad_function"]
            all_rows.append(page)
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(out_path, index=False)
        print(f"Fetched {len(out)} candidate rows, wrote {out_path}")
    else:
        print("No results returned from UniProt.")


# ---------- Stage 3: quality + dedup filter ----------

def _normalize_species(name):
    import re
    if pd.isna(name): return name
    name = re.sub(r"\s*\(strain[^)]*\)", "", name)
    name = re.sub(r"\s*\([^)]+\)", "", name).strip()
    return name


def stage_filter():
    raw = pd.read_csv(ROUND2 / "candidates_raw.csv")
    merged = pd.read_csv(HERE / "merged_dataset.csv")
    existing = set(merged["Entry"])

    # Standardize columns (UniProt TSV uses slightly different headers)
    rename_map = {
        "Entry": "Entry",
        "Accession": "Entry",
        "Entry Name": "EntryName",
        "Protein names": "protein_name",
        "Gene Names": "gene_names",
        "Organism": "Organism",
        "Organism (ID)": "organism_id",
        "Lineage": "lineage",
        "Taxonomic lineage": "lineage",
        "Taxonomic lineage (Ids)": "lineage_ids",
        "Length": "Length",
        "Sequence": "sequence",
        "EC number": "ec",
        "Keywords": "keywords",
        "Function [CC]": "cc_function",
        "Subcellular location [CC]": "cc_subcellular_location",
        "Subunit structure [CC]": "cc_subunit",
        "Pfam": "Pfam",
        "Protein families": "protein_families_raw",
    }
    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    before = len(raw)
    # Dedup by Entry (the same UniProt entry may match multiple cells)
    raw = raw.drop_duplicates(subset=["Entry"], keep="first")
    print(f"After dedup: {len(raw)} (was {before})")

    # Length filter (defensive - query already restricted)
    raw = raw[(raw["Length"] >= MIN_LEN) & (raw["Length"] <= MAX_LEN)]
    print(f"After length filter: {len(raw)}")

    # Exclude entries already in merged dataset
    raw = raw[~raw["Entry"].isin(existing)]
    print(f"After existing-dataset exclusion: {len(raw)}")

    # Strip out entries with non-canonical residues (defensive)
    def _ok_seq(s):
        if not isinstance(s, str) or len(s) == 0: return False
        bad = set(s.upper()) - set("ACDEFGHIKLMNPQRSTVWY")
        return len(bad) == 0 or bad == {"U"} or bad == {"X"} or bad <= {"U", "X"}
    raw = raw[raw["sequence"].apply(_ok_seq)]
    print(f"After sequence-validity filter: {len(raw)}")

    # Normalized species name
    raw["species"] = raw["Organism"].apply(_normalize_species)

    # Map UniProt lineage to a domain
    def _domain(line):
        if not isinstance(line, str): return None
        for d in ("Viruses", "Archaea", "Bacteria", "Eukaryota"):
            if d in line:
                return d
        return None
    raw["domain"] = raw["lineage"].apply(_domain) if "lineage" in raw.columns else None

    raw.to_csv(ROUND2 / "candidates_filtered.csv", index=False)
    print(f"Wrote {ROUND2/'candidates_filtered.csv'}: {len(raw)} candidates")


# ---------- Stage 4: download AF structures ----------

def stage_structures(limit=None):
    df = pd.read_csv(ROUND2 / "candidates_filtered.csv")
    if limit:
        df = df.head(limit)
    existing = scan_existing_structures(str(PDB_CACHE))
    status = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="AF download"):
        entry = row["Entry"]
        path = existing.get(entry)
        if path and Path(path).exists():
            status.append({"Entry": entry, "ok": True, "path": str(path), "cached": True})
            continue
        try:
            result = download_alphafold_structure(entry, str(PDB_CACHE))
            ok = result is not None and Path(result).exists()
            status.append({"Entry": entry, "ok": ok,
                           "path": str(result) if ok else "", "cached": False})
        except Exception as e:
            status.append({"Entry": entry, "ok": False, "path": "",
                           "error": str(e), "cached": False})
        time.sleep(0.05)  # polite
    s = pd.DataFrame(status)
    s.to_csv(ROUND2 / "structure_status.csv", index=False)
    print(f"AF success: {s['ok'].sum()}/{len(s)}")


# ---------- Stage 5: compute sequence + structural features ----------

def stage_features():
    df = pd.read_csv(ROUND2 / "candidates_filtered.csv")
    status = pd.read_csv(ROUND2 / "structure_status.csv")
    ok_status = status[status["ok"]].copy()
    path_by_entry = dict(zip(ok_status["Entry"], ok_status["path"]))
    have_struct = set(path_by_entry)
    df = df[df["Entry"].isin(have_struct)].copy()
    print(f"Computing features for {len(df)} entries with AF structures")

    # sequence features
    seq_feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Seq features"):
        f = calculate_sequence_features(str(row["sequence"]))
        f["Entry"] = row["Entry"]
        seq_feats.append(f)
    seq_df = pd.DataFrame(seq_feats)

    # structural features
    struct_feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Struct features"):
        path = path_by_entry.get(row["Entry"])
        try:
            f = extract_features(row["Entry"], str(path))
            if f is None:
                f = {"Entry": row["Entry"], "_err": "feature extraction failed"}
            struct_feats.append(f)
        except Exception as e:
            struct_feats.append({"Entry": row["Entry"], "_err": str(e)})
    struct_df = pd.DataFrame(struct_feats)

    out = df.merge(seq_df, on="Entry", how="left", suffixes=("", "_seq"))
    out = out.merge(struct_df, on="Entry", how="left", suffixes=("", "_struct"))

    # pLDDT filter
    if "avg_plddt" in out.columns:
        before = len(out)
        out = out[out["avg_plddt"].fillna(0) >= MIN_PLDDT]
        print(f"pLDDT≥{MIN_PLDDT} filter: kept {len(out)}/{before}")

    out.to_csv(ROUND2 / "features.csv", index=False)
    print(f"Wrote {ROUND2/'features.csv'}: {len(out)} entries")


# ---------- Stage 6: classify (protein_family, broad_function) ----------

def stage_classify():
    df = pd.read_csv(ROUND2 / "features.csv")
    # The classifier expects: Pfam, EC number, Protein families, etc.
    # Rename to what classify() expects
    rename = {
        "Pfam": "Pfam",
        "ec": "EC number",
        "protein_families_raw": "Protein families",
        "Organism": "Organism",
        "protein_name": "Protein names",
    }
    work = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    classified = classify(work)

    # Bring back the renamed columns and keep classifier output
    out = df.copy()
    for col in ("protein_family", "broad_function", "protein_name_clean",
                "is_enzyme", "is_transmembrane", "is_glycosylated", "has_disordered"):
        if col in classified.columns:
            out[col] = classified[col].values

    out.to_csv(ROUND2 / "classified.csv", index=False)
    print(f"Classified {len(out)} entries; wrote {ROUND2/'classified.csv'}")


# ---------- Stage 7: finalize - apply per-cell quotas and write ready-for-scoring CSV ----------

def stage_finalize():
    df = pd.read_csv(ROUND2 / "classified.csv")
    gap = pd.read_csv(HERE / "expansion_round2_gap_target.csv")
    targets = {(r["domain"], r["broad_function"]): int(r["add_target"])
               for _, r in gap.iterrows()}

    # Drop ribosomal/translation_factor that slipped past the keyword filter
    drop_bf = {"ribosomal", "translation_factor"}
    before = len(df)
    df = df[~df["broad_function"].isin(drop_bf)]
    print(f"Dropped {before - len(df)} entries reclassified as ribosomal/translation_factor")

    # Match cells using target_domain / target_broad_function (the intended cell
    # at fetch time). The classifier may have reassigned `broad_function`; the
    # `target_*` columns preserve the original cell intent for quota accounting.
    keep = []
    for (d, f), tgt in targets.items():
        sub = df[(df["target_domain"] == d) & (df["target_broad_function"] == f)].copy()
        if sub.empty:
            print(f"  [{d}/{f}] 0 candidates, target {tgt} - UNMET")
            continue
        # Per-species cap so a single species doesn't dominate this cell
        per_sp_cap = max(2, tgt // 20)
        sub = sub.groupby("species", group_keys=False).head(per_sp_cap)
        sub = sub.head(tgt)
        keep.append(sub)
        print(f"  [{d}/{f}] kept {len(sub)}/{tgt}")
    if not keep:
        print("Nothing to write.")
        return
    final = pd.concat(keep, ignore_index=True).drop_duplicates("Entry")

    # Drop Viruses: most have no AlphaFold model, and the few that slipped through
    # would skew the structure-conditioned analyses.
    before = len(final)
    final = final[final["target_domain"] != "Viruses"]
    print(f"Dropped {before - len(final)} viral entries (AFDB coverage is too sparse)")

    # Backfill `domain` from `target_domain` (UniProt lineage parse missed it)
    final["domain"] = final["target_domain"]
    final["source"] = "expansion_round2"
    final["structure_source"] = "AF"

    out = ROUND2 / "expansion_round2_for_scoring.csv"
    final.to_csv(out, index=False)
    print(f"\nFinal round-2 expansion: {len(final)} proteins")
    print(f"  by domain: {final['domain'].value_counts().to_dict()}")
    print(f"  by broad_function (top 10): "
          f"{final['broad_function'].value_counts().head(10).to_dict()}")
    print(f"  unique species: {final['species'].nunique()}")
    print(f"  ribosomal proteins (should be 0): {(final['broad_function']=='ribosomal').sum()}")
    print(f"Wrote {out}")


# ---------- Entry point ----------

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
    ap.add_argument("--stage", required=True,
                    choices=list(STAGES) + ["all"],
                    help="Which stage to run (or 'all')")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview without making network calls (where supported)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap entries processed (debugging)")
    args = ap.parse_args()

    if args.stage == "all":
        for name in ["queries", "fetch", "filter", "structures",
                     "features", "classify", "finalize"]:
            print(f"\n===== Stage: {name} =====")
            fn = STAGES[name]
            kwargs = {}
            if name == "queries" or name == "fetch":
                kwargs["dry_run"] = args.dry_run
            if name == "structures" and args.limit is not None:
                kwargs["limit"] = args.limit
            fn(**kwargs)
    else:
        fn = STAGES[args.stage]
        kwargs = {}
        if args.stage in ("queries", "fetch"):
            kwargs["dry_run"] = args.dry_run
        if args.stage == "structures" and args.limit is not None:
            kwargs["limit"] = args.limit
        fn(**kwargs)


if __name__ == "__main__":
    main()
