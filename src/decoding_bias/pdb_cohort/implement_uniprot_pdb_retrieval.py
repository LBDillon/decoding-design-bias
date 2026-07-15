"""Implement UniProt/PDB/AlphaFold recovery for the filter-C scored cohort.

This script operationalizes the retrieval recipe in
dataset_update/uniprot_pdb_missingness/report.md:

1. Batch fetch detailed UniProt metadata and cross references.
2. Fill missing UniProt-derived columns and regenerate missing classifier fields.
3. Fill alias/planning columns where the value is directly inferable.
4. Recover AlphaFold structure paths and pLDDT-derived breakdown columns.
5. Write a new enriched CSV plus diagnostics; the input CSV is not modified.

Usage:
    python dataset_update/implement_uniprot_pdb_retrieval.py

The default output is:
    dataset_update/main_plus_r2_r3_scored_filterC_v8.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dataset_update.protein_classification import classify  # noqa: E402
from decoding_bias.features.structural_features import (  # noqa: E402
    calculate_avg_cb_distance,
    calculate_compactness,
    calculate_contact_order,
    calculate_surface_exposure,
    download_alphafold_structure,
    extract_plddt_scores,
    extract_secondary_structure,
    parse_structure,
    scan_existing_structures,
)


DEFAULT_INPUT = HERE / "main_plus_r2_r3_scored_filterC_v7.csv"
DEFAULT_OUTPUT = HERE / "main_plus_r2_r3_scored_filterC_v8.csv"
DEFAULT_OUTDIR = HERE / "retrieval_recipe_outputs"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

UNIPROT_FIELDS = ",".join([
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
    "xref_pdb",
    "ft_transmem",
    "ft_carbohyd",
    "xref_disprot",
    "xref_ideal",
    "go_f",
    "xref_alphafolddb",
])

UNIPROT_RENAME = {
    "Entry Name": "EntryName",
    "Protein names": "protein_name",
    "Gene Names": "gene_names",
    "Organism": "Organism",
    "Organism (ID)": "organism_id",
    "Taxonomic lineage": "Taxonomic lineage",
    "Length": "Length",
    "Sequence": "sequence",
    "EC number": "ec",
    "Keywords": "keywords",
    "Function [CC]": "cc_function",
    "Subcellular location [CC]": "cc_subcellular_location",
    "Subunit structure": "Subunit structure",
    "Pfam": "Pfam",
    "Protein families": "protein_families_raw",
    "PDB": "pdb_ids_raw",
    "Transmembrane": "Transmembrane",
    "Glycosylation": "Glycosylation",
    "DisProt": "DisProt",
    "IDEAL": "IDEAL",
    "Gene Ontology (molecular function)": "Gene Ontology (molecular function)",
    "AlphaFoldDB": "AlphaFoldDB",
}

DIRECT_UNIPROT_FILL_COLUMNS = [
    "EntryName",
    "protein_name",
    "gene_names",
    "Organism",
    "organism_id",
    "Taxonomic lineage",
    "Length",
    "sequence",
    "ec",
    "keywords",
    "cc_function",
    "cc_subcellular_location",
    "Subunit structure",
    "Pfam",
    "protein_families_raw",
    "pdb_ids_raw",
]

CLASSIFIER_FILL_COLUMNS = [
    "protein_name_clean",
    "protein_family",
    "broad_function",
    "is_enzyme",
    "is_transmembrane",
    "is_glycosylated",
    "has_disordered",
    "has_pdb",
]

STRUCTURE_DERIVED_COLUMNS = [
    "avg_plddt",
    "min_plddt",
    "max_plddt",
    "plddt_very_high_pct",
    "plddt_high_pct",
    "plddt_medium_pct",
    "plddt_low_pct",
    "surface_exposure",
    "helix_percent",
    "sheet_percent",
    "loop_percent",
    "helix_sheet_contrast",
    "ordered_percent",
    "avg_cb_distance",
    "compactness",
    "structural_compactness",
    "centralization",
    "rco",
]

NULL_STRINGS = {"", "nan", "NaN", "None", "NA", "N/A", "<NA>"}
PDB_ID_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")
RANK_RE = re.compile(r"\s*([^,]+?)\s+\(([^)]+)\)\s*")
DOMAIN_NAMES = ("Viruses", "Archaea", "Bacteria", "Eukaryota")


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().isin(NULL_STRINGS)


def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in NULL_STRINGS:
        return True
    return False


def fill_missing_from_series(
    df: pd.DataFrame,
    values: pd.Series,
    column: str,
    counts: dict[str, int],
) -> None:
    """Fill only missing target values from a series indexed by Entry."""
    if column not in df.columns:
        incoming_nonmissing = values.dropna()
        sample = incoming_nonmissing.iloc[0] if not incoming_nonmissing.empty else None
        dtype = "object" if isinstance(sample, (str, bool, np.bool_)) else "float64"
        df[column] = pd.Series([np.nan] * len(df), index=df.index, dtype=dtype)
    elif not pd.api.types.is_object_dtype(df[column].dtype):
        incoming_nonmissing = values.dropna()
        if not incoming_nonmissing.empty:
            sample = incoming_nonmissing.iloc[0]
            if isinstance(sample, (str, bool, np.bool_)):
                df[column] = df[column].astype("object")
    incoming = df["Entry"].map(values)
    fill = missing_mask(df[column]) & incoming.notna() & ~incoming.astype("string").str.strip().isin(NULL_STRINGS)
    df.loc[fill, column] = incoming.loc[fill]
    counts[column] = counts.get(column, 0) + int(fill.sum())


def batch_fetch_uniprot(
    entries: list[str],
    batch_size: int,
    sleep_s: float,
    retries: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch UniProt TSV rows for accessions in batches."""
    rows = []
    failures = []
    session = requests.Session()

    for start in tqdm(range(0, len(entries), batch_size), desc="UniProt batches"):
        batch = entries[start:start + batch_size]
        query = " OR ".join(f"accession:{entry}" for entry in batch)
        params = {
            "query": query,
            "format": "tsv",
            "fields": UNIPROT_FIELDS,
            "size": len(batch),
        }

        response = None
        error = ""
        for attempt in range(1, retries + 1):
            try:
                response = session.get(UNIPROT_SEARCH, params=params, timeout=60)
                response.raise_for_status()
                break
            except Exception as exc:  # noqa: BLE001 - retry network/API hiccups.
                error = str(exc)
                response = None
                time.sleep(min(2 * attempt, 10))

        if response is None:
            for entry in batch:
                failures.append({"Entry": entry, "reason": error or "request_failed"})
            continue

        page = pd.read_csv(StringIO(response.text), sep="\t")
        if len(page):
            rows.append(page)
            got = set(page["Entry"].astype(str))
        else:
            got = set()
        for entry in batch:
            if entry not in got:
                failures.append({"Entry": entry, "reason": "not_returned_by_uniprot"})
        time.sleep(sleep_s)

    uni = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(uni):
        uni = uni.drop_duplicates("Entry", keep="first")
    return uni, pd.DataFrame(failures)


def prepare_uniprot_for_merge(uni_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize UniProt column names and compute classifier-derived fields."""
    uni = uni_raw.rename(columns={k: v for k, v in UNIPROT_RENAME.items() if k in uni_raw.columns}).copy()

    classifier_input = uni_raw.rename(columns={
        "Protein names": "Protein names",
        "Protein families": "Protein families",
        "EC number": "EC number",
        "Transmembrane": "Transmembrane",
        "Glycosylation": "Glycosylation",
        "PDB": "PDB",
        "DisProt": "DisProt",
        "IDEAL": "IDEAL",
        "Gene Ontology (molecular function)": "Gene Ontology (molecular function)",
    })
    classified = classify(classifier_input)
    classifier_keep = ["Entry"] + [c for c in CLASSIFIER_FILL_COLUMNS if c in classified.columns]
    classifier_df = classified[classifier_keep].copy()

    return uni, classifier_df


def parse_lineage_ranks(lineage: Any) -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "domain": None,
        "phylum_division": None,
        "class": None,
        "genus": None,
    }
    if not isinstance(lineage, str) or not lineage.strip():
        return out

    for part in lineage.split(","):
        match = RANK_RE.match(part)
        if not match:
            continue
        name, rank = match.group(1).strip(), match.group(2).strip().lower()
        if rank == "domain" and name in DOMAIN_NAMES:
            out["domain"] = name
        elif rank in {"phylum", "division"} and out["phylum_division"] is None:
            out["phylum_division"] = name
        elif rank == "class" and out["class"] is None:
            out["class"] = name
        elif rank == "genus" and out["genus"] is None:
            out["genus"] = name
    return out


def normalize_species(name: Any) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    cleaned = re.sub(r"\s*\(strain[^)]*\)", "", name)
    cleaned = re.sub(r"\s*\([^)]+\)", "", cleaned).strip()
    return cleaned or None


def first_pdb_id(raw_pdb: Any) -> str | None:
    if not isinstance(raw_pdb, str):
        return None
    match = PDB_ID_RE.search(raw_pdb)
    return match.group(0).upper() if match else None


def construct_description(row: pd.Series) -> str | None:
    entry = row.get("Entry")
    entry_name = row.get("EntryName")
    protein_name = row.get("protein_name")
    organism = row.get("Organism")
    organism_id = row.get("organism_id")
    gene_names = row.get("gene_names")
    if is_missing(entry) or is_missing(protein_name):
        return None
    prefix = f"sp|{entry}|{entry_name}" if not is_missing(entry_name) else str(entry)
    parts = [prefix, str(protein_name)]
    if not is_missing(organism):
        parts.append(f"OS={organism}")
    if not is_missing(organism_id):
        try:
            ox = str(int(float(organism_id)))
        except Exception:  # noqa: BLE001
            ox = str(organism_id)
        parts.append(f"OX={ox}")
    if not is_missing(gene_names):
        first_gene = str(gene_names).split()[0]
        parts.append(f"GN={first_gene}")
    return " ".join(parts)


def apply_uniprot_fills(
    df: pd.DataFrame,
    uni: pd.DataFrame,
    classifier_df: pd.DataFrame,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    uni_by_entry = uni.set_index("Entry", drop=False)

    for col in DIRECT_UNIPROT_FILL_COLUMNS:
        if col in uni_by_entry.columns:
            fill_missing_from_series(df, uni_by_entry[col], col, counts)

    # Alias Taxonomic lineage <-> lineage across rounds.
    if "Taxonomic lineage" in df.columns:
        fill_missing_from_series(df, df.set_index("Entry")["Taxonomic lineage"], "lineage", counts)
    if "lineage" in df.columns:
        fill_missing_from_series(df, df.set_index("Entry")["lineage"], "Taxonomic lineage", counts)

    # Taxonomy ranks and species from fetched organism/lineage.
    lineage_source = df["Taxonomic lineage"].where(~missing_mask(df["Taxonomic lineage"]), df.get("lineage"))
    ranks = lineage_source.apply(parse_lineage_ranks).apply(pd.Series)
    for col in ["domain", "phylum_division", "class", "genus"]:
        if col in ranks.columns:
            fill_missing_from_series(df, pd.Series(ranks[col].values, index=df["Entry"]), col, counts)

    if "Organism" in df.columns:
        species = df["Organism"].apply(normalize_species)
        fill_missing_from_series(df, pd.Series(species.values, index=df["Entry"]), "species", counts)

    # Description is a legacy display field, so construct it when absent.
    desc = df.apply(construct_description, axis=1)
    fill_missing_from_series(df, pd.Series(desc.values, index=df["Entry"]), "Description", counts)

    # Classifier-derived columns.
    classifier_by_entry = classifier_df.set_index("Entry", drop=False)
    for col in CLASSIFIER_FILL_COLUMNS:
        if col in classifier_by_entry.columns:
            fill_missing_from_series(df, classifier_by_entry[col], col, counts)

    # Experimental PDB availability from the fetched raw xref list.
    if "pdb_ids_raw" in df.columns:
        first_ids = df["pdb_ids_raw"].apply(first_pdb_id)
        fill_missing_from_series(df, pd.Series(first_ids.values, index=df["Entry"]), "pdb_id", counts)
        has_pdb = df["pdb_ids_raw"].notna() & ~missing_mask(df["pdb_ids_raw"])
        if "has_pdb" not in df.columns:
            df["has_pdb"] = np.nan
        fill = missing_mask(df["has_pdb"]) & has_pdb
        df.loc[fill, "has_pdb"] = True
        counts["has_pdb"] = counts.get("has_pdb", 0) + int(fill.sum())

        # `has_pdb_struct` was initialized as False for expansion rows in the
        # combine script, so this field needs a logical update rather than a
        # missing-only fill.
        if "has_pdb_struct" not in df.columns:
            df["has_pdb_struct"] = False
        before = df["has_pdb_struct"].fillna(False).astype(bool)
        after = before | has_pdb
        changed = after & ~before
        df.loc[changed, "has_pdb_struct"] = True
        counts["has_pdb_struct"] = counts.get("has_pdb_struct", 0) + int(changed.sum())

    # Uniform cell-label aliases for main and opposite expansion rounds.
    alias_pairs = [
        ("target_domain", "domain"),
        ("target_broad_function", "broad_function"),
        ("target_protein_family", "protein_family"),
    ]
    for target, source in alias_pairs:
        if source in df.columns:
            fill_missing_from_series(df, df.set_index("Entry")[source], target, counts)

    return counts


def load_structure_map(cache_dirs: list[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cache_dir in cache_dirs:
        for entry, path in scan_existing_structures(str(cache_dir)).items():
            mapping.setdefault(str(entry), path)
    return mapping


def structure_columns_missing(row: pd.Series) -> list[str]:
    return [c for c in STRUCTURE_DERIVED_COLUMNS if c in row.index and is_missing(row[c])]


def fill_cheap_structure_derivatives(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}

    derived = {
        "loop_percent": 1 - df["helix_percent"] - df["sheet_percent"]
        if {"helix_percent", "sheet_percent"}.issubset(df.columns) else None,
        "helix_sheet_contrast": df["helix_percent"] - df["sheet_percent"]
        if {"helix_percent", "sheet_percent"}.issubset(df.columns) else None,
        "ordered_percent": df["helix_percent"] + df["sheet_percent"]
        if {"helix_percent", "sheet_percent"}.issubset(df.columns) else None,
        "structural_compactness": 1 / df["compactness"]
        if "compactness" in df.columns else None,
        "centralization": 1 / df["avg_cb_distance"]
        if "avg_cb_distance" in df.columns else None,
    }

    for col, values in derived.items():
        if values is None:
            continue
        if col not in df.columns:
            df[col] = np.nan
        values = values.replace([np.inf, -np.inf], np.nan)
        fill = missing_mask(df[col]) & values.notna()
        df.loc[fill, col] = values.loc[fill]
        counts[col] = int(fill.sum())

    return counts


def extract_needed_structure_features(row: pd.Series, pdb_path: str) -> dict[str, Any]:
    """Compute only missing structure features for one row."""
    needed = set(structure_columns_missing(row))
    if not needed:
        return {}

    structure = parse_structure(pdb_path)
    if structure is None:
        return {"_structure_error": "parse_failed"}

    feats: dict[str, Any] = {}

    plddt_cols = {
        "avg_plddt",
        "min_plddt",
        "max_plddt",
        "plddt_very_high_pct",
        "plddt_high_pct",
        "plddt_medium_pct",
        "plddt_low_pct",
    }
    if needed & plddt_cols:
        feats.update(extract_plddt_scores(structure))

    ss_cols = {
        "helix_percent",
        "sheet_percent",
        "loop_percent",
        "helix_sheet_contrast",
        "ordered_percent",
    }
    if needed & ss_cols:
        feats.update(extract_secondary_structure(structure))

    if "surface_exposure" in needed:
        feats["surface_exposure"] = calculate_surface_exposure(structure)
    if "avg_cb_distance" in needed or "centralization" in needed:
        feats["avg_cb_distance"] = calculate_avg_cb_distance(structure)
    if "compactness" in needed or "structural_compactness" in needed:
        feats["compactness"] = calculate_compactness(structure)
    if "rco" in needed:
        feats["rco"] = calculate_contact_order(structure)

    if "compactness" in feats and not is_missing(feats["compactness"]) and feats["compactness"] > 0:
        feats["structural_compactness"] = 1.0 / feats["compactness"]
    if "avg_cb_distance" in feats and not is_missing(feats["avg_cb_distance"]) and feats["avg_cb_distance"] > 0:
        feats["centralization"] = 1.0 / feats["avg_cb_distance"]

    return feats


def apply_structure_recovery(
    df: pd.DataFrame,
    cache_dirs: list[Path],
    download_missing: bool,
    max_rows: int | None,
) -> tuple[dict[str, int], pd.DataFrame]:
    counts = fill_cheap_structure_derivatives(df)
    structure_map = load_structure_map(cache_dirs)
    status_rows = []

    if "pdb_path" not in df.columns:
        df["pdb_path"] = np.nan

    path_from_cache = df["Entry"].map(structure_map)
    fill_path = missing_mask(df["pdb_path"]) & path_from_cache.notna()
    df.loc[fill_path, "pdb_path"] = path_from_cache.loc[fill_path]
    counts["pdb_path"] = counts.get("pdb_path", 0) + int(fill_path.sum())

    needs = df[
        missing_mask(df["pdb_path"])
        | df.apply(lambda row: bool(structure_columns_missing(row)), axis=1)
    ].copy()
    if max_rows is not None:
        needs = needs.head(max_rows)

    for idx, row in tqdm(needs.iterrows(), total=len(needs), desc="Structure recovery"):
        entry = str(row["Entry"])
        path = row.get("pdb_path")
        downloaded = False
        error = ""

        if is_missing(path):
            if download_missing:
                try:
                    path = download_alphafold_structure(entry, str(HERE / "alphafold_cache"))
                    downloaded = bool(path)
                except Exception as exc:  # noqa: BLE001
                    path = None
                    error = str(exc)
            if path:
                df.at[idx, "pdb_path"] = path
                counts["pdb_path"] = counts.get("pdb_path", 0) + 1

        computed_cols: list[str] = []
        if path:
            refreshed = df.loc[idx].copy()
            refreshed["pdb_path"] = path
            try:
                feats = extract_needed_structure_features(refreshed, str(path))
                error = feats.pop("_structure_error", error)
                for col, val in feats.items():
                    if col not in df.columns:
                        df[col] = np.nan
                    if col in STRUCTURE_DERIVED_COLUMNS and is_missing(df.at[idx, col]) and not is_missing(val):
                        df.at[idx, col] = val
                        counts[col] = counts.get(col, 0) + 1
                        computed_cols.append(col)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        status_rows.append({
            "Entry": entry,
            "source": row.get("source"),
            "had_path_initially": not is_missing(row.get("pdb_path")),
            "path": path or "",
            "downloaded": downloaded,
            "computed_columns": ";".join(computed_cols),
            "error": error,
        })

    return counts, pd.DataFrame(status_rows)


def write_summary(
    outdir: Path,
    input_path: Path,
    output_path: Path,
    n_rows: int,
    uni_raw: pd.DataFrame,
    uni_failures: pd.DataFrame,
    fill_counts: dict[str, int],
    structure_counts: dict[str, int],
    structure_status: pd.DataFrame,
) -> None:
    lines = [
        "# Retrieval Recipe Implementation",
        "",
        f"Input: `{input_path}`",
        f"Output: `{output_path}`",
        f"Rows: {n_rows:,}",
        "",
        "## UniProt Fetch",
        f"- Returned rows: {len(uni_raw):,}",
        f"- Missing/not returned: {len(uni_failures):,}",
        "",
        "## Filled UniProt/Derived Columns",
    ]
    for col, n in sorted(fill_counts.items(), key=lambda x: (-x[1], x[0])):
        if n:
            lines.append(f"- {col}: {n:,}")
    lines.extend(["", "## Filled Structure Columns"])
    for col, n in sorted(structure_counts.items(), key=lambda x: (-x[1], x[0])):
        if n:
            lines.append(f"- {col}: {n:,}")
    if not structure_status.empty:
        lines.extend([
            "",
            "## Structure Recovery Status",
            f"- Rows attempted: {len(structure_status):,}",
            f"- Downloaded AlphaFold structures: {int(structure_status['downloaded'].sum()):,}",
            f"- Rows with errors: {int(structure_status['error'].astype(str).str.len().gt(0).sum()):,}",
        ])
    lines.extend([
        "",
        "## Diagnostics",
        "- `uniprot_fetch.tsv`",
        "- `uniprot_fetch_failures.csv`",
        "- `fill_counts.json`",
        "- `structure_recovery_status.csv`",
    ])
    (outdir / "retrieval_recipe_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--uniprot-batch-size", type=int, default=100)
    parser.add_argument("--uniprot-sleep", type=float, default=0.10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reuse-uniprot-fetch", type=Path, default=None,
                        help="Optional existing UniProt TSV to reuse instead of fetching.")
    parser.add_argument("--no-download-structures", action="store_true")
    parser.add_argument("--max-structure-rows", type=int, default=None,
                        help="Debug cap for structure recovery rows.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)
    entries = sorted(df["Entry"].astype(str).unique())

    print(f"Loaded {len(df):,} rows from {args.input}")
    if args.reuse_uniprot_fetch is not None:
        print(f"Reusing UniProt metadata from {args.reuse_uniprot_fetch}")
        uni_raw = pd.read_csv(args.reuse_uniprot_fetch, sep="\t", low_memory=False)
        returned = set(uni_raw["Entry"].astype(str)) if "Entry" in uni_raw.columns else set()
        uni_failures = pd.DataFrame(
            [{"Entry": entry, "reason": "not_in_reused_uniprot_fetch"}
             for entry in entries if entry not in returned]
        )
    else:
        print(f"Fetching UniProt metadata for {len(entries):,} unique accessions...")
        uni_raw, uni_failures = batch_fetch_uniprot(
            entries,
            batch_size=args.uniprot_batch_size,
            sleep_s=args.uniprot_sleep,
            retries=args.retries,
        )
    if uni_failures.empty:
        uni_failures = pd.DataFrame(columns=["Entry", "reason"])
    uni_raw.to_csv(args.outdir / "uniprot_fetch.tsv", sep="\t", index=False)
    uni_failures.to_csv(args.outdir / "uniprot_fetch_failures.csv", index=False)

    print(f"Preparing UniProt fills ({len(uni_raw):,} fetched rows)...")
    uni, classifier_df = prepare_uniprot_for_merge(uni_raw)
    fill_counts = apply_uniprot_fills(df, uni, classifier_df)

    print("Recovering structure paths and missing structure-derived columns...")
    cache_dirs = [
        ROOT / "data" / "pdbs_pifold_downloaded",
        HERE / "alphafold_cache",
        HERE / "pdb_cache",
    ]
    structure_counts, structure_status = apply_structure_recovery(
        df,
        cache_dirs=cache_dirs,
        download_missing=not args.no_download_structures,
        max_rows=args.max_structure_rows,
    )
    structure_status.to_csv(args.outdir / "structure_recovery_status.csv", index=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    (args.outdir / "fill_counts.json").write_text(json.dumps({
        "uniprot_and_derived": fill_counts,
        "structure": structure_counts,
    }, indent=2, sort_keys=True))
    write_summary(
        args.outdir,
        args.input,
        args.output,
        len(df),
        uni_raw,
        uni_failures,
        fill_counts,
        structure_counts,
        structure_status,
    )

    print(f"Wrote enriched CSV: {args.output}")
    print(f"Wrote diagnostics: {args.outdir}")


if __name__ == "__main__":
    main()
