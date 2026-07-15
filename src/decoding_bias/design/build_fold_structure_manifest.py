"""
Join downloaded ColabFold rank-1 PDBs back to all_designs_and_wt.csv.

The design table uses ids like:
    <uniprot>__<model>__d1

The folding FASTA/ColabFold output uses ids like:
    <uniprot>__<model>__s0
    <uniprot>__WT

This script records that mapping, adds the matching rank-1 PDB path to each
design/WT row, and optionally creates a flat symlink directory of the rank-1
PDBs so extract_design_features.py can consume them directly.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def rel_to_design(path: Path, *, resolve: bool = False) -> str:
    p = path.resolve() if resolve else path
    return os.path.relpath(p, HERE)


def fold_id_from_pdb_name(name: str) -> str | None:
    for marker in ("_unrelaxed_rank_001", "_relaxed_rank_001", "_rank_001"):
        if marker in name:
            return name.split(marker, 1)[0]
    return None


def expected_fold_id(row: pd.Series) -> str:
    if bool(row["is_wt"]):
        return f"{row['uniprot_id']}__WT"
    sample_idx = int(row["design_number"]) - 1
    return f"{row['uniprot_id']}__{row['model']}__s{sample_idx}"


def scan_rank1_pdbs(roots: list[Path]) -> tuple[dict[str, Path], pd.DataFrame]:
    """Return one preferred PDB per fold_id, plus a full scan table."""
    rows = []
    preferred: dict[str, Path] = {}

    for root in roots:
        root = root.expanduser()
        if not root.is_absolute():
            root = HERE / root
        if not root.exists():
            continue
        for pdb in sorted(root.rglob("*rank_001*.pdb")):
            fold_id = fold_id_from_pdb_name(pdb.name)
            if fold_id is None:
                continue
            resolved = pdb.resolve()
            rows.append(
                {
                    "fold_id": fold_id,
                    "pdb_path": rel_to_design(resolved),
                    "pdb_abs_path": str(resolved),
                    "pdb_file": pdb.name,
                    "source_root": rel_to_design(root),
                }
            )
            preferred.setdefault(fold_id, resolved)

    scan = pd.DataFrame(rows)
    return preferred, scan


def make_flat_links(mapping: dict[str, Path], flat_dir: Path) -> dict[str, Path]:
    flat_dir.mkdir(parents=True, exist_ok=True)
    linked: dict[str, Path] = {}
    for fold_id, target in sorted(mapping.items()):
        link = flat_dir / target.name
        if link.exists() or link.is_symlink():
            if link.resolve() == target.resolve():
                linked[fold_id] = link
                continue
            link.unlink()
        rel_target = os.path.relpath(target.resolve(), flat_dir.resolve())
        link.symlink_to(rel_target)
        linked[fold_id] = link
    return linked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-csv", default="outputs/all_designs_and_wt.csv")
    ap.add_argument("--fold-manifest", default="outputs/all_to_fold.manifest.csv")
    ap.add_argument(
        "--pdb-roots",
        nargs="+",
        default=[
            "arc_downloads/colabfold_chunks_home",
            "arc_downloads/colabfold_retry",
            "arc_downloads/colabfold_chunks",
            "arc_downloads/rank001_all",
        ],
        help="Downloaded rank-1 PDB directories, relative to design/ unless absolute.",
    )
    ap.add_argument("--flat-dir", default="arc_downloads/rank001_flat")
    ap.add_argument("--out", default="outputs/all_designs_and_wt_with_folds.csv")
    ap.add_argument("--structure-manifest", default="outputs/colabfold_rank1_manifest.csv")
    args = ap.parse_args()

    design_csv = HERE / args.design_csv
    fold_manifest = HERE / args.fold_manifest
    out = HERE / args.out
    structure_manifest = HERE / args.structure_manifest
    flat_dir = HERE / args.flat_dir

    designs = pd.read_csv(design_csv)
    folds = pd.read_csv(fold_manifest)

    designs["fold_id"] = designs.apply(expected_fold_id, axis=1)
    manifest_ids = set(folds["fold_id"])
    missing_from_manifest = sorted(set(designs["fold_id"]) - manifest_ids)
    if missing_from_manifest:
        raise SystemExit(
            f"{len(missing_from_manifest)} derived fold_ids are absent from {fold_manifest}: "
            f"{missing_from_manifest[:5]}"
        )

    pdb_roots = [Path(p) for p in args.pdb_roots]
    pdb_by_fold, scan = scan_rank1_pdbs(pdb_roots)
    links_by_fold = make_flat_links(pdb_by_fold, flat_dir)

    designs["has_colabfold_rank1"] = designs["fold_id"].isin(pdb_by_fold)
    designs["colabfold_rank1_pdb"] = designs["fold_id"].map(
        lambda fid: rel_to_design(pdb_by_fold[fid]) if fid in pdb_by_fold else ""
    )
    designs["colabfold_rank1_pdb_abs"] = designs["fold_id"].map(
        lambda fid: str(pdb_by_fold[fid]) if fid in pdb_by_fold else ""
    )
    designs["colabfold_rank1_flat_pdb"] = designs["fold_id"].map(
        lambda fid: rel_to_design(links_by_fold[fid]) if fid in links_by_fold else ""
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    designs.to_csv(out, index=False)

    if scan.empty:
        scan = pd.DataFrame(columns=["fold_id", "pdb_path", "pdb_abs_path", "pdb_file", "source_root"])
    scan["is_preferred"] = scan.apply(
        lambda r: r["fold_id"] in pdb_by_fold and r["pdb_abs_path"] == str(pdb_by_fold[r["fold_id"]]),
        axis=1,
    )
    scan.to_csv(structure_manifest, index=False)

    n_expected = len(designs)
    n_found = int(designs["has_colabfold_rank1"].sum())
    n_designs = int((~designs["is_wt"]).sum())
    n_designs_found = int((designs["has_colabfold_rank1"] & ~designs["is_wt"]).sum())
    n_wt = int(designs["is_wt"].sum())
    n_wt_found = int((designs["has_colabfold_rank1"] & designs["is_wt"]).sum())
    dupes = int(scan.duplicated("fold_id").sum()) if not scan.empty else 0

    print(f"Expected rows: {n_expected}")
    print(f"Rank-1 structures linked: {n_found}/{n_expected}")
    print(f"  designs: {n_designs_found}/{n_designs}")
    print(f"  WTs:     {n_wt_found}/{n_wt}")
    print(f"Duplicate rank-1 PDB entries seen: {dupes}")
    print(f"Wrote: {out}")
    print(f"Wrote: {structure_manifest}")
    print(f"Flat symlinks: {flat_dir} ({len(links_by_fold)} links)")


if __name__ == "__main__":
    main()
