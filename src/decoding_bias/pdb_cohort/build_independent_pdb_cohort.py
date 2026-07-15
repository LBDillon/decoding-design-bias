"""Build an INDEPENDENT, non-redundant experimental-PDB cohort to test whether
the taxonomic-bias results replicate (reviewer R3.4).

Unlike the paired cohort (design/prep_pdb_inputs_fresh.py), this does NOT start
from our dataset's UniProt accessions. It draws fresh from the whole PDB:

  X-ray, <=2.5 A, monomeric (1 polymer-entity instance), protein, length 50-1000,
  from Bacteria / Eukaryota / Archaea  --> RCSB Search API
    -> one representative per 30% sequence-identity cluster (non-redundant)
    -> RCSB Data API (GraphQL): chain sequence, UniProt mapping, organism, resolution
    -> UniProt fields -> protein_classification.classify (protein_family, broad_function)
    -> collapse_species_subspecies.collapse (merge subspecies/serovar labels)
    -> domain from lineage
    -> breadth filter (family >=5 species, species >=2) on the cohort's own taxonomy
    -> seeded subsample to the MAIN dataset's domain marginal (function carried as
       a covariate, NOT hard-matched: ribosomal is best-effort by design)

Stages (resumable; each writes to design/outputs/independent_cohort/):
  search    candidates_representatives.csv   (entity ids + group counts)
  annotate  annotated.csv                    (seq, uniprot, org, domain, classification)
  match     cohort_manifest.csv + scoring inputs

Usage:
  python design/build_independent_pdb_cohort.py search
  python design/build_independent_pdb_cohort.py annotate
  python design/build_independent_pdb_cohort.py match
  python design/build_independent_pdb_cohort.py all          # run all three
Options: --resolution 2.5 --min-len 50 --max-len 1000 --identity 30
         --seed 0 --target-n 0 (0 = max at domain marginal)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dataset_update.protein_classification import classify  # noqa: E402
from dataset_update.collapse_species_subspecies import collapse  # noqa: E402

OUT = HERE / "outputs" / "independent_cohort"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Reference composition target = the updated paper's dataset = full v12 metadata
# (main + round2 + round3, n=10,148).
MAIN_META = ROOT / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"

DOMAINS = ["Bacteria", "Eukaryota", "Archaea"]

# UniProt TSV fields required by protein_classification.classify(). The TSV
# header names (left of the rename) are what classify() reads, so we keep them.
UNIPROT_FIELDS = ",".join([
    "accession", "protein_name", "protein_families", "xref_pfam", "ec",
    "go_f", "ft_transmem", "ft_carbohyd", "xref_pdb", "xref_disprot",
    "xref_ideal", "lineage", "organism_name", "length",
])
# Map UniProt TSV headers -> the column names classify() expects.
CLASSIFY_RENAME = {
    "Protein names": "Protein names",
    "Protein families": "Protein families",
    "Pfam": "Pfam",
    "EC number": "EC number",
    "Gene Ontology (molecular function)": "Gene Ontology (molecular function)",
    "Transmembrane": "Transmembrane",
    "Glycosylation": "Glycosylation",
    "PDB": "PDB",
    "DisProt": "DisProt",
    "IDEAL": "IDEAL",
}

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _get_json(url: str, retries: int = 4, timeout: int = 90) -> dict:
    last = None
    for attempt in range(1, retries + 1):
        try:
            return json.load(urllib.request.urlopen(url, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code == 204:  # no content = empty result page
                return {}
            last = e
            time.sleep(min(2 * attempt, 12))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 * attempt, 12))
    raise RuntimeError(f"GET failed after {retries} tries: {last}")


def _post_json(url: str, payload: dict, retries: int = 4, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 * attempt, 12))
    raise RuntimeError(f"POST failed after {retries} tries: {last}")


def domain_from_lineage(lineage: str | None) -> str | None:
    """First superkingdom token found in a UniProt 'Taxonomic lineage' string."""
    if not isinstance(lineage, str):
        return None
    for d in DOMAINS:
        if re.search(rf"\b{d}\b", lineage):
            return d
    return None


# --------------------------------------------------------------------------- #
# stage 1: search
# --------------------------------------------------------------------------- #
def build_query(args, start: int, rows: int) -> dict:
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "exptl.method", "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": args.resolution}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match", "value": "Protein"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "range", "value": {
                    "from": args.min_len, "to": args.max_len,
                    "include_lower": True, "include_upper": True}}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                "operator": "in", "value": DOMAINS}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_assembly_info.polymer_entity_instance_count",
                "operator": "equals", "value": 1}},
        ]},
        "return_type": "polymer_entity",
        "request_options": {
            "group_by": {"aggregation_method": "sequence_identity",
                         "similarity_cutoff": args.identity},
            "group_by_return_type": "representatives",
            "paginate": {"start": start, "rows": rows},
        },
    }


def stage_search(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_per = 1000
    # first call: get counts
    q = build_query(args, 0, rows_per)
    url = SEARCH_URL + "?json=" + urllib.parse.quote(json.dumps(q))
    out = _get_json(url)
    total = out.get("total_count")
    n_groups = out.get("group_by_count")
    print(f"matching entities: {total:,}  |  30%-id representatives: {n_groups:,}")

    ids = [r["identifier"] for r in out.get("result_set", [])]
    start = rows_per
    while start < n_groups:
        q = build_query(args, start, rows_per)
        url = SEARCH_URL + "?json=" + urllib.parse.quote(json.dumps(q))
        page = _get_json(url)
        batch = [r["identifier"] for r in page.get("result_set", [])]
        if not batch:
            break
        ids.extend(batch)
        start += rows_per
        print(f"  collected {len(ids):,}/{n_groups:,} representatives", end="\r")
        time.sleep(0.1)
    print()
    df = pd.DataFrame({"entity_id": ids})
    df.to_csv(OUT / "candidates_representatives.csv", index=False)
    print(f"wrote {OUT / 'candidates_representatives.csv'}  ({len(df):,} rows)")


# --------------------------------------------------------------------------- #
# stage 2: annotate
# --------------------------------------------------------------------------- #
GRAPHQL_Q = """
{ polymer_entities(entity_ids: %s) {
    rcsb_id
    entity_poly { pdbx_seq_one_letter_code_can rcsb_sample_sequence_length }
    rcsb_polymer_entity { pdbx_description }
    rcsb_polymer_entity_container_identifiers { entry_id uniprot_ids auth_asym_ids }
    rcsb_entity_source_organism { ncbi_scientific_name ncbi_taxonomy_id }
    entry { rcsb_entry_info { resolution_combined } }
} }"""


def fetch_graphql(entity_ids: list[str], batch: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(0, len(entity_ids), batch):
        chunk = entity_ids[i:i + batch]
        payload = {"query": GRAPHQL_Q % json.dumps(chunk)}
        out = _post_json(GRAPHQL_URL, payload)
        for e in (out.get("data", {}).get("polymer_entities") or []):
            if e is None:
                continue
            ci = e.get("rcsb_polymer_entity_container_identifiers") or {}
            poly = e.get("entity_poly") or {}
            org = (e.get("rcsb_entity_source_organism") or [{}])[0] or {}
            entry = (e.get("entry") or {}).get("rcsb_entry_info") or {}
            res = entry.get("resolution_combined")
            uni = ci.get("uniprot_ids") or []
            chains = ci.get("auth_asym_ids") or []
            rows.append({
                "entity_id": e["rcsb_id"],
                "pdb_id": ci.get("entry_id"),
                "chain": chains[0] if chains else None,
                "uniprot": uni[0] if uni else None,
                "sequence": poly.get("pdbx_seq_one_letter_code_can"),
                "Length": poly.get("rcsb_sample_sequence_length"),
                "pdb_description": (e.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                "organism_rcsb": org.get("ncbi_scientific_name"),
                "ncbi_taxid": org.get("ncbi_taxonomy_id"),
                "resolution_A": res[0] if isinstance(res, list) and res else res,
            })
        print(f"  graphql {min(i + batch, len(entity_ids)):,}/{len(entity_ids):,}", end="\r")
        time.sleep(0.05)
    print()
    return pd.DataFrame(rows)


def fetch_uniprot(accessions: list[str], batch: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(0, len(accessions), batch):
        chunk = accessions[i:i + batch]
        query = " OR ".join(f"accession:{a}" for a in chunk)
        params = urllib.parse.urlencode({
            "query": query, "format": "tsv", "fields": UNIPROT_FIELDS, "size": len(chunk)})
        try:
            txt = urllib.request.urlopen(f"{UNIPROT_SEARCH}?{params}", timeout=90).read().decode()
            page = pd.read_csv(StringIO(txt), sep="\t")
            if len(page):
                rows.append(page)
        except Exception as e:  # noqa: BLE001
            print(f"  uniprot batch {i} failed: {e}")
        print(f"  uniprot {min(i + batch, len(accessions)):,}/{len(accessions):,}", end="\r")
        time.sleep(0.1)
    print()
    uni = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(uni):
        uni = uni.drop_duplicates("Entry", keep="first")
    return uni


def stage_annotate(args) -> None:
    cand = pd.read_csv(OUT / "candidates_representatives.csv")
    ids = cand["entity_id"].tolist()
    gcache = OUT / "_graphql_raw.csv"
    if gcache.exists() and not args.refetch:
        print(f"reusing cached GraphQL ({gcache})")
        g = pd.read_csv(gcache)
    else:
        print(f"annotating {len(ids):,} representatives via GraphQL...")
        g = fetch_graphql(ids)
        g.to_csv(gcache, index=False)

    # standard-AA + length sanity (sequence is the resolved/sample chain seq)
    g = g[g["sequence"].notna()].copy()
    g["sequence"] = g["sequence"].str.upper().str.replace("U", "C", regex=False)
    g["nonstd"] = g["sequence"].apply(lambda s: bool(set(s) - STD_AA))
    g = g[~g["nonstd"]].copy()
    g = g[(g["Length"] >= args.min_len) & (g["Length"] <= args.max_len)]
    print(f"  {len(g):,} pass standard-AA + length")

    # UniProt fetch for mapped entities (for classification + lineage/domain)
    accs = sorted(g["uniprot"].dropna().unique().tolist())
    ucache = OUT / "_uniprot_raw.tsv"
    if ucache.exists() and not args.refetch:
        print(f"reusing cached UniProt ({ucache})")
        uni = pd.read_csv(ucache, sep="\t", low_memory=False)
    else:
        print(f"fetching UniProt fields for {len(accs):,} accessions...")
        uni = fetch_uniprot(accs)
        uni.to_csv(ucache, sep="\t", index=False)

    # classify on the UniProt rows, then map back onto entities by accession
    cls_in = uni.rename(columns=CLASSIFY_RENAME)
    for col in CLASSIFY_RENAME.values():
        if col not in cls_in.columns:
            cls_in[col] = pd.NA
    classified = classify(cls_in)[
        ["Entry", "protein_name_clean", "protein_family", "broad_function",
         "is_enzyme", "is_transmembrane", "is_glycosylated", "has_disordered"]
    ].rename(columns={"Entry": "uniprot"})

    # domain + species from UniProt lineage/organism (authoritative for matching)
    lin = uni.rename(columns={"Taxonomic lineage": "lineage"}) if "Taxonomic lineage" in uni else uni
    lin = lin[["Entry", "lineage", "Organism"]].rename(
        columns={"Entry": "uniprot", "Organism": "organism_uniprot"})
    lin["domain"] = lin["lineage"].apply(domain_from_lineage)

    df = g.merge(classified, on="uniprot", how="left").merge(lin, on="uniprot", how="left")

    # species: prefer UniProt organism, else RCSB organism; then collapse subspecies
    df["species"] = df["organism_uniprot"].fillna(df["organism_rcsb"])
    df["species"] = df["species"].apply(
        lambda x: re.sub(r"\s*\([^)]*\)", "", x).strip() if isinstance(x, str) else x)
    df["species_collapsed"] = df["species"].apply(collapse)

    # domain fallback: entities with no UniProt mapping get classified 'other'
    df["broad_function"] = df["broad_function"].fillna("other")
    df["protein_family"] = df["protein_family"].fillna("Unclassified")

    n_nodomain = df["domain"].isna().sum()
    df = df[df["domain"].notna()].copy()
    print(f"  dropped {n_nodomain:,} without resolvable domain; {len(df):,} annotated")

    df.to_csv(OUT / "annotated.csv", index=False)
    print(f"wrote {OUT / 'annotated.csv'}  ({len(df):,} rows)")
    print("\ndomain distribution (annotated pool):")
    print(df["domain"].value_counts())
    print("\nbroad_function (top 12):")
    print(df["broad_function"].value_counts().head(12))


# --------------------------------------------------------------------------- #
# stage 3: match
# --------------------------------------------------------------------------- #
def breadth_filter(df: pd.DataFrame, min_species_per_family: int = 5,
                   min_proteins_per_species: int = 2) -> pd.DataFrame:
    """Iteratively enforce: each species has >=2 proteins AND each protein_family
    is represented by >=5 distinct species, within the cohort's own taxonomy."""
    cur = df.copy()
    while True:
        sp_counts = cur.groupby("species_collapsed")["entity_id"].transform("count")
        cur = cur[sp_counts >= min_proteins_per_species]
        fam_species = cur.groupby("protein_family")["species_collapsed"].transform("nunique")
        cur2 = cur[fam_species >= min_species_per_family]
        if len(cur2) == len(cur):
            return cur2
        cur = cur2


def stage_match(args) -> None:
    df = pd.read_csv(OUT / "annotated.csv")
    print(f"annotated pool: {len(df):,}")

    df = breadth_filter(df)
    print(f"after breadth filter (family>=5 species, species>=2): {len(df):,}")
    print(df["domain"].value_counts())

    # target = the paper's dataset domain marginal = FULL v12 (all sources)
    meta = pd.read_csv(MAIN_META, low_memory=False)
    target = meta["domain"].value_counts(normalize=True)
    print("\ntarget (full v12) domain marginal:")
    print(target.round(3))

    avail = df["domain"].value_counts()
    # largest N such that every domain's required count is available
    max_n = min(int(avail.get(d, 0) / target.get(d, 1e-9)) for d in target.index if target.get(d, 0) > 0)
    n = args.target_n if args.target_n > 0 else max_n
    n = min(n, max_n)
    print(f"\nmax cohort at exact domain marginal: {max_n:,}  -> building N={n:,}")

    parts = []
    rng_seed = args.seed
    for d, frac in target.items():
        k = int(round(frac * n))
        sub = df[df["domain"] == d]
        k = min(k, len(sub))
        parts.append(sub.sample(n=k, random_state=rng_seed))
    cohort = pd.concat(parts, ignore_index=True)
    cohort.to_csv(OUT / "cohort_manifest.csv", index=False)
    print(f"\nwrote {OUT / 'cohort_manifest.csv'}  ({len(cohort):,} proteins)")
    print("cohort domain mix:")
    print((cohort["domain"].value_counts(normalize=True)).round(3))
    print("cohort broad_function (top 12; ribosomal is best-effort):")
    print(cohort["broad_function"].value_counts().head(12))

    # scoring input: sequence file (PDB chain sequence) + structure fetch list
    cohort[["entity_id", "pdb_id", "chain", "uniprot", "sequence", "Length",
            "domain", "species_collapsed", "protein_family", "broad_function",
            "resolution_A"]].to_csv(OUT / "cohort_scoring_inputs.csv", index=False)
    cohort[["pdb_id", "chain", "entity_id"]].drop_duplicates().to_csv(
        OUT / "cohort_structures_to_download.csv", index=False)
    print(f"wrote cohort_scoring_inputs.csv + cohort_structures_to_download.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["search", "annotate", "match", "all"])
    ap.add_argument("--resolution", type=float, default=2.5)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--identity", type=int, default=30,
                    choices=[30, 40, 50, 70, 90, 95, 100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-n", type=int, default=0,
                    help="0 = max possible at exact domain marginal")
    ap.add_argument("--refetch", action="store_true",
                    help="ignore cached GraphQL/UniProt files and re-download")
    args = ap.parse_args()

    if args.stage in ("search", "all"):
        stage_search(args)
    if args.stage in ("annotate", "all"):
        stage_annotate(args)
    if args.stage in ("match", "all"):
        stage_match(args)


if __name__ == "__main__":
    main()
