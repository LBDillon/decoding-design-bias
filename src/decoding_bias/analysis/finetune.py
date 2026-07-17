"""Reproduction for the fine-tuning surface-chemistry results.

Model training and sequence generation require GPUs and checkpoints.  The deposited
table contains one row per generated design (plus matched WT rows), which is enough
to refit the surface acid-base PCA and recompute the protein-level paired tests in
main Tables 5-6.  Table S22 uses the separate matched-secretome observations.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SURFACE_FEATURES = [
    "surface_acidic_fraction", "surface_basic_fraction",
    "surface_ionizable_fraction", "surface_net_charge",
]
DIRECT_FEATURES = [
    "surface_net_charge", "surface_acidic_fraction", "surface_basic_fraction",
    "isoelectric_point", "charge_per_residue", "surface_ionizable_fraction",
]
WT_NAMES = {"WT", "WildType", "wildtype", "Wild-type"}
COMPARISONS = [
    ("AlkalineMPNN_020", "ProteinMPNN"),
    ("AcidophileMPNN_020", "ProteinMPNN"),
    ("AlkalineMPNN", "ProteinMPNN_v002"),
    ("AcidophileMPNN", "ProteinMPNN_v002"),
]
SELF_CONSISTENCY_MODELS = [
    "ProteinMPNN_v020(base)", "AlkSecMPNN_020", "AcidSecMPNN_020",
    "WT_singleseq(control)",
]


def _holm(values: np.ndarray) -> np.ndarray:
    """Holm family-wise adjustment in original row order."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    out = np.empty_like(adjusted_ranked)
    out[order] = np.minimum(adjusted_ranked, 1.0)
    return out


def _wilcoxon(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    # Matches R wilcox.test(..., exact=FALSE): normal approximation with correction.
    return float(stats.wilcoxon(values, method="approx", correction=True).pvalue)


def _surface_pca(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    complete = table.dropna(subset=SURFACE_FEATURES).copy()
    matrix = complete[SURFACE_FEATURES].to_numpy(float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=1)
    standardized = (matrix - center) / scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    rotation = vt.T
    scores = standardized @ rotation[:, :2]
    acid_index = SURFACE_FEATURES.index("surface_acidic_fraction")
    if rotation[acid_index, 0] < 0:
        rotation[:, 0] *= -1
        scores[:, 0] *= -1
    projected = complete[["uniprot_id", "model"]].copy()
    projected["PC1"] = scores[:, 0]
    projected["PC2"] = scores[:, 1]
    loadings = pd.DataFrame({"feature": SURFACE_FEATURES, "PC1": rotation[:, 0], "PC2": rotation[:, 1]})
    explained = singular ** 2 / np.sum(singular ** 2)
    return projected, loadings, explained


def surface_shift_tests(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    projected, loadings, explained = _surface_pca(table)
    wt = projected[projected["model"].isin(WT_NAMES)].drop_duplicates("uniprot_id").set_index("uniprot_id")
    designs = projected[~projected["model"].isin(WT_NAMES)].copy()
    designs = designs.join(wt[["PC1", "PC2"]].rename(columns={"PC1": "PC1_wt", "PC2": "PC2_wt"}),
                           on="uniprot_id", how="inner")
    per_protein = (
        designs.groupby(["uniprot_id", "model"], observed=True)
        .agg(n_designs=("PC1", "size"), PC1_mean=("PC1", "mean"), PC2_mean=("PC2", "mean"),
             PC1_wt=("PC1_wt", "first"), PC2_wt=("PC2_wt", "first"))
        .reset_index()
    )
    per_protein["dPC1"] = per_protein["PC1_mean"] - per_protein["PC1_wt"]
    per_protein["dPC2"] = per_protein["PC2_mean"] - per_protein["PC2_wt"]

    rows = []
    for arm, base in COMPARISONS:
        wide = per_protein[per_protein["model"].isin([arm, base])].pivot(
            index="uniprot_id", columns="model", values=["dPC1", "dPC2"]
        )
        if ("dPC1", arm) not in wide or ("dPC1", base) not in wide:
            continue
        d1 = (wide[("dPC1", arm)] - wide[("dPC1", base)]).dropna().to_numpy(float)
        d2 = (wide[("dPC2", arm)] - wide[("dPC2", base)]).dropna().to_numpy(float)
        rows.append({
            "comparison": f"{arm} minus {base}",
            "n_PC1": len(d1), "mean_delta_PC1": d1.mean(), "median_delta_PC1": np.median(d1),
            "wilcox_p_PC1": _wilcoxon(d1),
            "n_PC2": len(d2), "mean_delta_PC2": d2.mean(), "median_delta_PC2": np.median(d2),
            "wilcox_p_PC2": _wilcoxon(d2),
        })
    tests = pd.DataFrame(rows)
    tests["p_adj_PC1"] = _holm(tests["wilcox_p_PC1"].to_numpy())
    tests["p_adj_PC2"] = _holm(tests["wilcox_p_PC2"].to_numpy())
    loadings["explained_variance"] = [explained[0], explained[1], np.nan, np.nan]
    return per_protein, tests, loadings


def direct_feature_tests(table: pd.DataFrame) -> pd.DataFrame:
    wt = table[table["model"].isin(WT_NAMES)].drop_duplicates("uniprot_id").set_index("uniprot_id")
    designs = table[~table["model"].isin(WT_NAMES)].copy()
    rows = []
    for feature in DIRECT_FEATURES:
        designs[f"_delta_{feature}"] = designs[feature] - designs["uniprot_id"].map(wt[feature])
    per_protein = designs.groupby(["uniprot_id", "model"], observed=True)[
        [f"_delta_{feature}" for feature in DIRECT_FEATURES]
    ].mean()

    for arm, base in COMPARISONS:
        for feature in DIRECT_FEATURES:
            values = per_protein[f"_delta_{feature}"].unstack("model")
            if arm not in values or base not in values:
                continue
            delta = (values[arm] - values[base]).dropna().to_numpy(float)
            rows.append({
                "feature": feature,
                "comparison": f"{arm} minus {base}",
                "n": len(delta),
                "mean_delta_vs_base": delta.mean(),
                "median_delta_vs_base": np.median(delta),
                "wilcox_p": _wilcoxon(delta),
            })
    result = pd.DataFrame(rows)
    result["p_adj_holm"] = np.nan
    for _, indices in result.groupby("comparison", sort=False).groups.items():
        result.loc[indices, "p_adj_holm"] = _holm(result.loc[indices, "wilcox_p"].to_numpy())
    return result


def table_s22(source: Path) -> pd.DataFrame:
    matched = pd.read_csv(source)
    rows = []
    for family, group in matched.groupby("family", observed=True):
        values = group["case_minus_control"].to_numpy(float)
        rows.append({
            "model": family,
            "mean_steer": values.mean(),
            "sem": values.std(ddof=1) / np.sqrt(len(values)),
            "n_targets": len(values),
            "n_acidic": int(np.sum(values < 0)),
        })
    return pd.DataFrame(rows)


def self_consistency(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise refold-to-input scTM and paired fine-tuned-vs-base tests."""
    per_protein = (
        table.groupby(["model", "uniprot_id"], observed=True)
        [["scTM", "scRMSD", "pLDDT"]].mean().reset_index()
    )
    summary_rows = []
    for model in SELF_CONSISTENCY_MODELS:
        raw = table[table["model"] == model]
        group = per_protein[per_protein["model"] == model]
        summary_rows.append({
            "model": model,
            "n_structures": len(raw),
            "n_proteins": group["uniprot_id"].nunique(),
            "mean_scTM": group["scTM"].mean(),
            "median_scTM": group["scTM"].median(),
            "mean_scRMSD": group["scRMSD"].mean(),
            "mean_pLDDT": group["pLDDT"].mean(),
        })

    base = per_protein[per_protein["model"] == "ProteinMPNN_v020(base)"].set_index("uniprot_id")
    test_rows = []
    for model in ["AlkSecMPNN_020", "AcidSecMPNN_020"]:
        fine_tuned = per_protein[per_protein["model"] == model].set_index("uniprot_id")
        shared = base.index.intersection(fine_tuned.index)
        for metric in ["scTM", "scRMSD", "pLDDT"]:
            delta = fine_tuned.loc[shared, metric] - base.loc[shared, metric]
            test_rows.append({
                "comparison": f"{model} minus ProteinMPNN_v020(base)",
                "metric": metric,
                "n": len(shared),
                "mean_delta": delta.mean(),
                "wilcoxon_p": stats.wilcoxon(
                    fine_tuned.loc[shared, metric], base.loc[shared, metric]
                ).pvalue,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def _plot(per_protein: pd.DataFrame, path: Path) -> None:
    models = ["ProteinMPNN", "AlkalineMPNN_020", "AcidophileMPNN_020"]
    colors = {"ProteinMPNN": "#7f8c8d", "AlkalineMPNN_020": "#c0392b", "AcidophileMPNN_020": "#2980b9"}
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for model in models:
        group = per_protein[per_protein["model"] == model]
        ax.scatter(group["dPC1"], group["dPC2"], s=14, alpha=.35, color=colors[model])
        mean = group[["dPC1", "dPC2"]].mean()
        ax.arrow(0, 0, mean["dPC1"], mean["dPC2"], color=colors[model], width=.015,
                 length_includes_head=True, label=model)
        ax.text(mean["dPC1"], mean["dPC2"], model, color=colors[model], fontsize=9)
    ax.axhline(0, color="grey", lw=.6, ls="--"); ax.axvline(0, color="grey", lw=.6, ls="--")
    ax.set_xlabel("surface acid-base PC1 shift (+ = more acidic)")
    ax.set_ylabel("surface acid-base PC2 shift")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def run(cfg, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out = Path(out_dir) if out_dir else cfg.stage_output("finetune")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(cfg.finetune_dir / "design_surface_features.csv")
    self_consistency_table = pd.read_csv(cfg.finetune_dir / "self_consistency.csv")
    per_protein, pca_tests, loadings = surface_shift_tests(table)
    direct = direct_feature_tests(table)
    matched = table_s22(cfg.finetune_dir / "surface_shift_matched.csv")
    sc_summary, sc_tests = self_consistency(self_consistency_table)

    per_protein.to_csv(out / "surface_pca_per_protein.csv", index=False)
    pca_tests.to_csv(out / "surface_pca_base_relative_tests.csv", index=False)
    loadings.to_csv(out / "surface_pca_loadings.csv", index=False)
    direct.to_csv(out / "direct_feature_base_relative_tests.csv", index=False)
    matched.to_csv(out / "table_s22_surface_steer.csv", index=False)
    sc_summary.to_csv(out / "self_consistency_summary.csv", index=False)
    sc_tests.to_csv(out / "self_consistency_paired_tests.csv", index=False)
    _plot(per_protein, out / "finetune_surface_shift.png")
    print(f"[finetune] Tables 5-6/S22 and scTM self-consistency -> {out}")
    return {"pca": pca_tests, "direct": direct, "matched": matched,
            "self_consistency": sc_summary, "self_consistency_tests": sc_tests,
            "per_protein": per_protein}
