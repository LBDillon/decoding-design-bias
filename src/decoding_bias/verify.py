"""Verification harness: re-run the reproducible stages and compare against the
committed reference artifacts in tests/reference/, plus a few headline paper
numbers. Backs `decoding-bias verify`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import repo_root

REF_DIR = repo_root() / "tests" / "reference"


def compare_csv(ref_path, new_path, *, rtol=1e-6, atol=1e-8, label="") -> tuple[bool, str]:
    ref = pd.read_csv(ref_path)
    new = pd.read_csv(new_path)
    msgs = []
    if list(ref.columns) != list(new.columns):
        msgs.append(f"columns differ ({len(ref.columns)} vs {len(new.columns)})")
    if len(ref) != len(new):
        msgs.append(f"rows differ ({len(ref)} vs {len(new)})")
    common = [c for c in ref.columns if c in new.columns]
    n = min(len(ref), len(new))
    max_abs = 0.0
    for c in common:
        rc, nc = ref[c].iloc[:n], new[c].iloc[:n]
        if pd.api.types.is_numeric_dtype(rc) and pd.api.types.is_numeric_dtype(nc):
            a, b = rc.to_numpy(float), nc.to_numpy(float)
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.where(both_nan, 0.0, np.abs(a - b))
            max_abs = max(max_abs, float(np.nanmax(diff)) if len(diff) else 0.0)
            bad = ~(np.isclose(a, b, rtol=rtol, atol=atol) | both_nan)
            if bad.any():
                msgs.append(f"col {c}: {int(bad.sum())} float mismatches (max {np.nanmax(diff):.2e})")
        else:
            if not rc.astype(str).equals(nc.astype(str)):
                msgs.append(f"col {c}: categorical mismatch")
    ok = not msgs
    detail = f"max_abs_diff={max_abs:.2e}" if ok else "; ".join(msgs)
    return ok, f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}"


def run(cfg, *, fast: bool = True) -> bool:
    """Re-run reproducible stages into a temp dir and diff vs tests/reference/."""
    from .analysis import variance, importance, pca
    from .data import load_analysis_table, dataset_composition

    results: list[tuple[bool, str]] = []
    tmp = Path(tempfile.mkdtemp(prefix="dbias_verify_"))

    # -- variance decomposition (Table 1/2) --
    variance.run(cfg, out_dir=tmp / "vd", make_figures=False)
    results.append(compare_csv(REF_DIR / "variance/score_variance_decomposition.csv",
                               tmp / "vd/score_variance_decomposition.csv",
                               label="variance decomposition (Table 1/2)"))

    # -- pLDDT-as-score decomposition (Table S13) --
    variance.run_plddt(cfg, out_dir=tmp / "plddt")
    results.append(compare_csv(REF_DIR / "variance/plddt_vd.csv",
                               tmp / "plddt/plddt_vd.csv",
                               label="pLDDT variance decomposition (Table S13)"))

    # -- property importance (SI S16-S18) --
    importance.run(cfg, out_dir=tmp / "imp")
    for f in ["property_importance_expanded.csv", "physchem_relweight_wide_expanded.csv",
              "physchem_relweight_wide_finetune020.csv"]:
        results.append(compare_csv(REF_DIR / "importance" / f, tmp / "imp" / f,
                                   label=f"importance/{f}"))

    # -- PCA loadings + compactness (Table S14) --
    pres = pca.run(cfg, out_dir=tmp / "pca")
    for f in ["pca_loadings.csv", "compactness_pairwise.csv"]:
        results.append(compare_csv(REF_DIR / "pca" / f, tmp / "pca" / f, label=f"pca/{f}"))
    # cross-validate the Python PCA against the deposited R/mgcv notebook (Fig 3, Table S14/S15)
    md = pca.crosscheck_gam(cfg, pres["compactness"], tmp / "pca")
    if md is not None:
        ok = md < 1e-3
        results.append((ok, f"[{'PASS' if ok else 'FAIL'}] Python PCA matches R notebook "
                            f"(compactness overlap diff {md:.2e}; Fig 3 / Table S14/S15 source)"))

    # -- headline paper numbers (spec check, not a reference file) --
    df = load_analysis_table(cfg.analysis_table)
    comp = dataset_composition(df)
    ok_n = comp["n_proteins"] == 10148 and comp["n_species"] == 495 and comp["n_families"] == 281
    results.append((ok_n, f"[{'PASS' if ok_n else 'FAIL'}] dataset composition (Table 7): "
                          f"{comp['n_proteins']} proteins / {comp['n_species']} species / "
                          f"{comp['n_families']} families (paper 10148/495/281)"))

    # -- fine-tuning surface steer (Table S22) - from the deposited matched-surface data --
    from .analysis import finetune
    fs = finetune.table_s22(cfg, out_dir=tmp / "finetune").set_index("model")
    ref_s22 = pd.read_csv(REF_DIR / "finetune/table_s22.csv").set_index("model")
    common = [m for m in ref_s22.index if m in fs.index]
    md = max(abs(float(ref_s22.loc[m, "mean_steer"]) - float(fs.loc[m, "mean_steer"]))
             for m in common) if common else float("nan")
    ok_s22 = len(common) == len(ref_s22) and md < 1e-9
    results.append((ok_s22, f"[{'PASS' if ok_s22 else 'FAIL'}] fine-tuning surface steer (Table S22): "
                            f"max |Δmean|={md:.2e} over {len(common)}/{len(ref_s22)} models"))

    # -- design analyses (Fig 4/S21 shifts + Table 4/S20 functional residue) - if design_dir set --
    if cfg.external("design_dir") is not None:
        from .analysis import design
        res = design.run(cfg, out_dir=tmp / "design")
        if res["shift"] is not None:
            mine = res["shift"].rename(columns={"property": "feature", "cohens_dz": "dz"})
            ref = pd.read_csv(REF_DIR / "design/table_s21_dz.csv")
            m2 = ref.merge(mine[["model", "feature", "dz"]], on=["model", "feature"], suffixes=("_ref", "_mine"))
            md = float((m2["dz_ref"] - m2["dz_mine"]).abs().max()) if len(m2) else float("nan")
            ok = len(m2) == len(ref) and md < 1e-6
            results.append((ok, f"[{'PASS' if ok else 'FAIL'}] design shift dz (Table S21): "
                                f"max |Δdz|={md:.2e} over {len(m2)}/{len(ref)} cells"))
        if res["functional"] is not None:
            ref = pd.read_csv(REF_DIR / "design/functional_residue_by_model.csv").set_index("model")
            mine = res["functional"].set_index("model")
            common = [m for m in ref.index if m in mine.index]
            md = max(abs(float(ref.loc[m, "delta"]) - float(mine.loc[m, "delta"])) for m in common) if common else float("nan")
            ok = len(common) == len(ref) and md < 1e-6
            results.append((ok, f"[{'PASS' if ok else 'FAIL'}] functional-residue Δ (Table 4/S20): "
                                f"max |Δ|={md:.2e} over {len(common)}/{len(ref)} models"))

    # -- Elo (slow; full only) --
    if not fast:
        from .analysis import elo
        elo.run(cfg, arms=("full",), weightings=("unweighted",),
                out_dir=tmp / "elo", make_paper_figures=False)
        results.append(compare_csv(
            REF_DIR / "elo/all_models_species_ratings_long.csv",
            tmp / "elo/elo_full_unweighted/results/all_models_species_ratings_long.csv",
            label="Elo species ratings (Fig 2 / Table 3)"))

    print("\n=== decoding-bias verify ===")
    for _, msg in results:
        print(" ", msg)
    passed = sum(ok for ok, _ in results)
    print(f"\n{passed}/{len(results)} checks passed"
          + ("  (fast mode: Elo skipped, use --full)" if fast else ""))
    return all(ok for ok, _ in results)
