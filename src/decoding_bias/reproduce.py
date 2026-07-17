"""One-command reviewer reproduction and reference comparison."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


REQUIRED_INPUTS = {
    "main score/feature table": "data/main_analysis.csv",
    "species taxonomy": "data/taxonomy.csv",
    "design features": "data/design/designs_features.csv",
    "wild-type design features": "data/design/wt_features.csv",
    "functional-site observations": "data/design/functional_residue_recovery.csv",
    "fine-tuning surface features": "data/finetune/design_surface_features.csv",
    "matched fine-tuning observations": "data/finetune/surface_shift_matched.csv",
    "fine-tuning self-consistency": "data/finetune/self_consistency.csv",
}


def missing_inputs(cfg: Config) -> list[str]:
    return [f"{label}: {cfg.root / path}" for label, path in REQUIRED_INPUTS.items()
            if not (cfg.root / path).exists()]


def _compare(
    expected: Path,
    actual: Path,
    *,
    label: str,
    keys: list[str] | None = None,
    columns: list[str] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> tuple[bool, str]:
    reference = pd.read_csv(expected)
    observed = pd.read_csv(actual)
    if keys:
        if any(key not in reference or key not in observed for key in keys):
            return False, f"{label}: missing comparison key(s)"
        reference = reference.sort_values(keys).reset_index(drop=True)
        observed = observed.sort_values(keys).reset_index(drop=True)
    if len(reference) != len(observed):
        return False, f"{label}: row count {len(observed)} != expected {len(reference)}"
    compare_columns = columns or list(reference.columns)
    missing = [column for column in compare_columns if column not in observed]
    if missing:
        return False, f"{label}: missing columns {missing}"
    max_difference = 0.0
    for column in compare_columns:
        left, right = reference[column], observed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(float), right.to_numpy(float)
            both_nan = np.isnan(a) & np.isnan(b)
            if not np.all(np.isclose(a, b, atol=atol, rtol=rtol) | both_nan):
                difference = np.abs(a - b)
                return False, f"{label}: numeric mismatch in {column} (max {np.nanmax(difference):.2e})"
            difference = np.abs(a - b)
            if np.isfinite(difference).any():
                max_difference = max(max_difference, float(np.nanmax(difference)))
        elif not left.fillna("").astype(str).equals(right.fillna("").astype(str)):
            return False, f"{label}: categorical mismatch in {column}"
    return True, f"{label}: max numeric difference {max_difference:.2e}"


def _record(rows: list[dict], claim: str, status: str, detail: str, seconds: float) -> None:
    rows.append({"claim": claim, "status": status, "seconds": round(seconds, 2), "detail": detail})


def _run_gam(cfg: Config, out: Path) -> tuple[bool, str]:
    rscript = shutil.which("Rscript")
    if not rscript:
        return False, "Rscript not found; install the R environment described in REPRODUCIBILITY.md"
    script = cfg.root / "src/decoding_bias/analysis/gam_landscapes.R"
    completed = subprocess.run(
        [rscript, str(script), str(cfg.analysis_table), str(out)],
        cwd=cfg.root,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        return False, (completed.stderr or completed.stdout).strip().splitlines()[-1]
    # Compare mappings shared with the manuscript-era R table.
    expected = pd.read_csv(cfg.gam_dir / "gam_deviance_reference.csv")
    actual = pd.read_csv(out / "gam_deviance.csv")
    expected = expected.rename(columns={"gam_dev_explained_pct": "value"})
    actual = actual.rename(columns={"gam_dev_explained_pct": "value"})
    shared = expected.merge(actual, on="model", suffixes=("_expected", "_actual"))
    difference = (shared["value_expected"] - shared["value_actual"]).abs()
    ok = len(shared) >= 10 and float(difference.max()) < .11
    detail = (f"{len(shared)} stable model mappings compared; max GAM deviance difference "
              f"{float(difference.max()):.3f} percentage points")
    return ok, detail


def run(cfg: Config | None = None, *, quick: bool = False) -> tuple[bool, pd.DataFrame]:
    """Reproduce the paper-facing analyses and write ``reproduction_report.md``.

    ``quick=True`` skips the seeded 14-model Elo run and R/mgcv landscapes.  All
    other main numeric results are still recomputed from row-level deposited data.
    """
    cfg = cfg or Config.load()
    missing = missing_inputs(cfg)
    if missing:
        raise FileNotFoundError("Missing reviewer inputs:\n  " + "\n  ".join(missing))

    from .analysis import design, elo, finetune, importance, pca, variance
    from .data import dataset_composition, load_analysis_table

    root = cfg.stage_output("reviewer")
    rows: list[dict] = []

    started = time.perf_counter()
    composition = dataset_composition(load_analysis_table(cfg.analysis_table))
    ok = (composition["n_proteins"], composition["n_species"], composition["n_families"]) == (10148, 495, 281)
    _record(rows, "Dataset composition", "PASS" if ok else "FAIL",
            f"{composition['n_proteins']} proteins / {composition['n_species']} species / "
            f"{composition['n_families']} families", time.perf_counter() - started)

    started = time.perf_counter()
    variance.run(cfg, out_dir=root / "variance", make_figures=True)
    variance.run_plddt(cfg, out_dir=root / "variance")
    checks = [
        _compare(cfg.expected_dir / "variance/score_variance_decomposition.csv",
                 root / "variance/score_variance_decomposition.csv", label="variance decomposition",
                 keys=["model"]),
        _compare(cfg.expected_dir / "variance/plddt_vd.csv", root / "variance/plddt_vd.csv",
                 label="pLDDT control", keys=["model"]),
    ]
    ok = all(value for value, _ in checks)
    _record(rows, "Variance decomposition (Tables 1-2)", "PASS" if ok else "FAIL",
            "; ".join(detail for _, detail in checks), time.perf_counter() - started)

    started = time.perf_counter()
    pca.run(cfg, out_dir=root / "pca")
    checks = [
        _compare(cfg.expected_dir / "pca/pca_loadings.csv", root / "pca/pca_loadings.csv",
                 label="PCA loadings", keys=["feature"]),
        _compare(cfg.expected_dir / "pca/compactness_pairwise.csv", root / "pca/compactness_pairwise.csv",
                 label="domain overlap", keys=["domain"]),
    ]
    ok = all(value for value, _ in checks)
    _record(rows, "Biophysical PCA (Figure 3A-B)", "PASS" if ok else "FAIL",
            "; ".join(detail for _, detail in checks), time.perf_counter() - started)

    started = time.perf_counter()
    importance.run(cfg, out_dir=root / "importance")
    checks = [
        _compare(cfg.expected_dir / "importance/property_importance_expanded.csv",
                 root / "importance/property_importance_expanded.csv", label="property importance",
                 keys=["group", "model", "property"]),
        _compare(cfg.expected_dir / "importance/physchem_relweight_wide_expanded.csv",
                 root / "importance/physchem_relweight_wide_expanded.csv", label="importance wide table",
                 keys=["property"]),
    ]
    ok = all(value for value, _ in checks)
    _record(rows, "Property importance (Tables S16-S18)", "PASS" if ok else "FAIL",
            "; ".join(detail for _, detail in checks), time.perf_counter() - started)

    started = time.perf_counter()
    design.run(cfg, out_dir=root / "design")
    checks = [
        _compare(cfg.expected_dir / "design/table_s21_dz.csv", root / "design/physchem_effect_sizes.csv",
                 label="design shifts", keys=["model", "feature"],
                 columns=["model", "feature", "n_templates", "mean_delta", "dz"]),
        _compare(cfg.expected_dir / "design/functional_residue_by_model.csv",
                 root / "design/functional_residue_conservation_by_model.csv",
                 label="functional-residue recovery", keys=["model"]),
    ]
    ok = all(value for value, _ in checks)
    _record(rows, "Designed-sequence shifts (Figure 4, Table 4)", "PASS" if ok else "FAIL",
            "; ".join(detail for _, detail in checks), time.perf_counter() - started)

    started = time.perf_counter()
    finetune.run(cfg, out_dir=root / "finetune")
    checks = [
        _compare(cfg.expected_dir / "finetune/ft_design_shift_surface_acid_base_paired_tests.csv",
                 root / "finetune/surface_pca_base_relative_tests.csv", label="surface PCA shifts",
                 keys=["comparison"]),
        _compare(cfg.expected_dir / "finetune/ft_design_direct_feature_base_relative_tests.csv",
                 root / "finetune/direct_feature_base_relative_tests.csv", label="direct feature shifts",
                 keys=["comparison", "feature"]),
        _compare(cfg.expected_dir / "finetune/table_s22.csv",
                 root / "finetune/table_s22_surface_steer.csv", label="matched surface steer",
                 keys=["model"]),
        _compare(cfg.expected_dir / "finetune/self_consistency_summary.csv",
                 root / "finetune/self_consistency_summary.csv", label="self-consistency summary",
                 keys=["model"]),
        _compare(cfg.expected_dir / "finetune/self_consistency_paired_tests.csv",
                 root / "finetune/self_consistency_paired_tests.csv", label="self-consistency tests",
                 keys=["comparison", "metric"]),
    ]
    ok = all(value for value, _ in checks)
    _record(rows, "Fine-tuning steer and scTM (Figure 5, Tables 5-6)", "PASS" if ok else "FAIL",
            "; ".join(detail for _, detail in checks), time.perf_counter() - started)

    if quick:
        _record(rows, "Species Elo (Figure 2, Table 3)", "SKIP", "run without --quick", 0)
        _record(rows, "GAM landscapes (Figure 3C)", "SKIP", "run without --quick", 0)
    else:
        started = time.perf_counter()
        elo.run(cfg, out_dir=root / "taxonomy", n_permutations=50, make_figure=True)
        ok, detail = _compare(
            cfg.expected_dir / "elo/all_models_species_ratings_long.csv",
            root / "taxonomy/all_models_species_ratings_long.csv",
            label="species Elo", keys=["model", "species"])
        _record(rows, "Species Elo (Figure 2, Table 3)", "PASS" if ok else "FAIL", detail,
                time.perf_counter() - started)

        started = time.perf_counter()
        ok, detail = _run_gam(cfg, root / "gam")
        _record(rows, "GAM landscapes (Figure 3C)", "PASS" if ok else "FAIL", detail,
                time.perf_counter() - started)

    report = pd.DataFrame(rows)
    report.to_csv(root / "reproduction_report.csv", index=False)
    lines = ["# Reproduction report", "", f"Mode: {'quick' if quick else 'full reviewer'}", "",
             "| Claim | Status | Seconds | Detail |", "|---|---:|---:|---|"]
    for row in report.itertuples(index=False):
        detail = str(row.detail).replace("|", "\\|")
        lines.append(f"| {row.claim} | {row.status} | {row.seconds:.2f} | {detail} |")
    (root / "reproduction_report.md").write_text("\n".join(lines) + "\n")
    passed = bool((report["status"] != "FAIL").all())
    print("\n" + report[["claim", "status", "seconds"]].to_string(index=False))
    print(f"\nReport: {root / 'reproduction_report.md'}")
    return passed, report
