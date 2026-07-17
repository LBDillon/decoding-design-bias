"""Reproduction for the designed-sequence results (Figure 4, Table 4).

The deposited inputs start after model generation and structure prediction:
per-design features, matched wild-type features, and per-design functional-site
recovery.  That is the smallest level of data that still lets a reviewer recompute
the paired effect sizes and statistical tests.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODELS = [
    "ProteinMPNN", "SolubleMPNN", "Caliby", "SolubleCaliby",
    "ESM-IF", "MIF", "MIF-ST",
]

# The twelve non-constant properties reported in manuscript Table S21.
PROPERTIES = [
    "mw_per_residue", "isoelectric_point", "acidic_residue_fraction",
    "basic_residue_fraction", "gravy", "aromaticity", "instability_index",
    "proline_fraction", "ordered_percent", "helix_sheet_contrast", "rco",
    "avg_cb_distance",
]


def _bh_adjust(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment in original row order."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def design_shifts(designs: pd.DataFrame, wild_types: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return protein-level deltas and model/property paired statistics."""
    designs = designs[designs["model"].isin(MODELS)].copy()
    wt = wild_types.set_index("uniprot_id")
    means = designs.groupby(["model", "uniprot_id", "domain"], observed=True)[PROPERTIES].mean()
    rows: list[dict] = []
    for (model, uid, domain), values in means.iterrows():
        if uid not in wt.index:
            continue
        row = {"model": model, "uniprot_id": uid, "domain": domain}
        row.update((values - wt.loc[uid, PROPERTIES]).to_dict())
        rows.append(row)
    delta = pd.DataFrame(rows)

    stats_rows: list[dict] = []
    for model in MODELS:
        group = delta[delta["model"] == model]
        for feature in PROPERTIES:
            values = group[feature].dropna().to_numpy(float)
            dz = values.mean() / values.std(ddof=1) if len(values) >= 3 and values.std(ddof=1) else np.nan
            p = stats.wilcoxon(values).pvalue if len(values) >= 3 and not np.allclose(values, 0) else np.nan
            stats_rows.append({
                "model": model,
                "feature": feature,
                "n_templates": len(values),
                "mean_delta": values.mean() if len(values) else np.nan,
                "dz": dz,
                "wilcoxon_p": p,
            })
    result = pd.DataFrame(stats_rows)
    mask = result["wilcoxon_p"].notna()
    result["p_fdr"] = np.nan
    result.loc[mask, "p_fdr"] = _bh_adjust(result.loc[mask, "wilcoxon_p"].to_numpy())
    return delta, result


def functional_residue_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate design-level recovery using proteins as the statistical unit."""
    per_protein = (
        observations.groupby(["model", "uniprot_id"], observed=True)
        [["func_recovery", "bg_recovery"]].mean().reset_index()
    )
    rows = []
    for model, group in per_protein.groupby("model", observed=True):
        difference = (group["func_recovery"] - group["bg_recovery"]).to_numpy(float)
        p = stats.wilcoxon(difference, alternative="two-sided").pvalue if len(difference) >= 5 else np.nan
        rows.append({
            "model": model,
            "func_rec": group["func_recovery"].mean(),
            "bg_rec": group["bg_recovery"].mean(),
            "delta": difference.mean(),
            "p": p,
            "n_proteins": len(group),
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def _plot_shifts(result: pd.DataFrame, path: Path) -> None:
    matrix = result.pivot(index="feature", columns="model", values="dz").reindex(PROPERTIES, columns=MODELS)
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=35, ha="right")
    ax.set_yticks(range(len(PROPERTIES)), [x.replace("_", " ") for x in PROPERTIES])
    fig.colorbar(image, ax=ax, label="paired Cohen's dz (design − WT)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_functional(result: pd.DataFrame, path: Path) -> None:
    result = result[result["model"].isin(MODELS)].set_index("model").reindex(MODELS).reset_index()
    x = np.arange(len(result)); width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, result["func_rec"], width, label="functional sites", color="#c9282d")
    ax.bar(x + width / 2, result["bg_rec"], width, label="background", color="#9f9f9f")
    ax.set_xticks(x, result["model"], rotation=35, ha="right")
    ax.set_ylabel("WT-sequence recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(cfg, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out = Path(out_dir) if out_dir else cfg.stage_output("design")
    out.mkdir(parents=True, exist_ok=True)
    designs = pd.read_csv(cfg.design_dir / "designs_features.csv")
    wild_types = pd.read_csv(cfg.design_dir / "wt_features.csv")
    functional = pd.read_csv(cfg.design_dir / "functional_residue_recovery.csv")

    delta, shifts = design_shifts(designs, wild_types)
    functional_summary = functional_residue_summary(functional)
    delta.to_csv(out / "wt_design_deltas.csv", index=False)
    shifts.to_csv(out / "physchem_effect_sizes.csv", index=False)
    functional_summary.to_csv(out / "functional_residue_conservation_by_model.csv", index=False)
    _plot_shifts(shifts, out / "physchem_shift_heatmap.png")
    _plot_functional(functional_summary, out / "functional_residue_conservation.png")
    print(f"[design] {len(MODELS)} models × {len(PROPERTIES)} properties; "
          f"{functional['uniprot_id'].nunique()} functional-site templates -> {out}")
    return {"shift": shifts, "functional": functional_summary, "delta": delta}
