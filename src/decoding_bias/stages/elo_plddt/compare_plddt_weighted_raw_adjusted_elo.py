"""Compare weighted, raw, and pLDDT-adjusted model Elo ratings.

This script separates two AF-quality effects:
  1. Weighting effect: pLDDT-weighted Elo minus unweighted/raw Elo.
  2. Baseline/confounding effect: raw Elo minus pLDDT-adjusted Elo.

The adjusted Elo is expected to be created by
dataset_update/control_model_elo_for_plddt_baseline.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTED = (
    REPO_ROOT
    / "outputs"
    / "elo_analysis_10models"
    / "results"
    / "all_models_species_ratings_long.csv"
)
DEFAULT_RAW = (
    REPO_ROOT
    / "dataset_update"
    / "kingdom_plddt"
    / "model_elo_unweighted"
    / "results"
    / "all_models_species_ratings_long.csv"
)
DEFAULT_ADJUSTED = (
    REPO_ROOT
    / "dataset_update"
    / "kingdom_plddt"
    / "plddt_controlled_model_elo"
    / "results"
    / "plddt_adjusted_species_ratings_long.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "dataset_update" / "kingdom_plddt" / "plddt_weighted_raw_adjusted_comparison"
)


def fmt_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value == 0:
        return "<1e-300"
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def load_comparison(weighted_path: Path, raw_path: Path, adjusted_path: Path) -> pd.DataFrame:
    weighted = pd.read_csv(weighted_path)[["species", "model", "rating"]].rename(
        columns={"rating": "weighted_rating"}
    )
    raw = pd.read_csv(raw_path)[["species", "model", "rating", "domain"]].rename(
        columns={"rating": "raw_rating"}
    )
    adjusted = pd.read_csv(adjusted_path)[
        [
            "species",
            "model",
            "plddt_elo",
            "lineage_kingdom",
            "n_entries",
            "mean_entry_plddt",
            "median_entry_plddt",
            "plddt_expected_rating",
            "plddt_adjusted_rating",
        ]
    ]

    df = raw.merge(weighted, on=["species", "model"], how="inner").merge(
        adjusted, on=["species", "model"], how="inner"
    )
    df["weighting_shift"] = df["weighted_rating"] - df["raw_rating"]
    df["plddt_expected_component"] = df["plddt_expected_rating"] - 1500.0
    df["plddt_control_shift"] = df["plddt_adjusted_rating"] - df["raw_rating"]
    df["raw_deviation"] = df["raw_rating"] - 1500.0
    df["weighted_deviation"] = df["weighted_rating"] - 1500.0
    df["adjusted_deviation"] = df["plddt_adjusted_rating"] - 1500.0
    return df


def domain_spread(df: pd.DataFrame, rating_col: str) -> pd.Series:
    means = df.groupby("domain")[rating_col].mean()
    return pd.Series(
        {
            f"{rating_col}_domain_spread": means.max() - means.min(),
            f"{rating_col}_archaea_mean": means.get("Archaea", np.nan),
            f"{rating_col}_bacteria_mean": means.get("Bacteria", np.nan),
            f"{rating_col}_eukaryota_mean": means.get("Eukaryota", np.nan),
        }
    )


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in df.groupby("model"):
        raw_weighted = stats.pearsonr(group["raw_rating"], group["weighted_rating"])
        raw_adjusted = stats.pearsonr(group["raw_rating"], group["plddt_adjusted_rating"])
        raw_plddt = stats.pearsonr(group["raw_rating"], group["plddt_elo"])
        weighted_plddt = stats.pearsonr(group["weighted_rating"], group["plddt_elo"])

        spread = pd.concat(
            [
                domain_spread(group, "weighted_rating"),
                domain_spread(group, "raw_rating"),
                domain_spread(group, "plddt_adjusted_rating"),
            ]
        )

        rows.append(
            {
                "model": model,
                "n_species": len(group),
                "weighted_raw_pearson_r": raw_weighted.statistic,
                "weighted_raw_mae": (group["weighting_shift"].abs()).mean(),
                "weighted_raw_rmse": np.sqrt((group["weighting_shift"] ** 2).mean()),
                "weighted_raw_max_abs_shift": group["weighting_shift"].abs().max(),
                "raw_plddt_pearson_r": raw_plddt.statistic,
                "raw_plddt_r_squared": raw_plddt.statistic**2,
                "weighted_plddt_pearson_r": weighted_plddt.statistic,
                "weighted_plddt_r_squared": weighted_plddt.statistic**2,
                "raw_adjusted_pearson_r": raw_adjusted.statistic,
                "mean_abs_plddt_expected_component": group[
                    "plddt_expected_component"
                ].abs().mean(),
                "rmse_plddt_expected_component": np.sqrt(
                    (group["plddt_expected_component"] ** 2).mean()
                ),
                "raw_sd": group["raw_rating"].std(),
                "weighted_sd": group["weighted_rating"].std(),
                "adjusted_sd": group["plddt_adjusted_rating"].std(),
                **spread.to_dict(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["weighting_spread_change"] = (
        summary["weighted_rating_domain_spread"] - summary["raw_rating_domain_spread"]
    )
    summary["control_spread_change"] = (
        summary["plddt_adjusted_rating_domain_spread"]
        - summary["raw_rating_domain_spread"]
    )
    summary["control_percent_domain_spread_remaining"] = (
        summary["plddt_adjusted_rating_domain_spread"]
        / summary["raw_rating_domain_spread"]
    )
    return summary.sort_values("raw_plddt_r_squared", ascending=False)


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(["model", group_col], dropna=False)
        .agg(
            n_species=("species", "size"),
            mean_weighted_rating=("weighted_rating", "mean"),
            mean_raw_rating=("raw_rating", "mean"),
            mean_plddt_adjusted_rating=("plddt_adjusted_rating", "mean"),
            mean_weighting_shift=("weighting_shift", "mean"),
            mean_abs_weighting_shift=("weighting_shift", lambda s: s.abs().mean()),
            mean_plddt_control_shift=("plddt_control_shift", "mean"),
            mean_abs_plddt_control_shift=(
                "plddt_control_shift",
                lambda s: s.abs().mean(),
            ),
            mean_plddt_expected_component=("plddt_expected_component", "mean"),
            mean_plddt_elo=("plddt_elo", "mean"),
            mean_species_avg_plddt=("mean_entry_plddt", "mean"),
        )
        .reset_index()
        .sort_values(["model", group_col])
    )


def plot_comparison(
    summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    model_order = summary["model"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    sns.barplot(
        data=summary,
        y="model",
        x="raw_plddt_r_squared",
        order=model_order,
        color="#4C78A8",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("How much raw Elo is predicted by pLDDT-only Elo")
    axes[0, 0].set_xlabel("R-squared")
    axes[0, 0].set_ylabel("")

    shift = summary.melt(
        id_vars="model",
        value_vars=["weighted_raw_mae", "mean_abs_plddt_expected_component"],
        var_name="effect",
        value_name="mean_abs_elo_points",
    )
    shift["effect"] = shift["effect"].map(
        {
            "weighted_raw_mae": "pLDDT weighting effect\n|weighted - raw|",
            "mean_abs_plddt_expected_component": "pLDDT baseline component\n|expected - 1500|",
        }
    )
    sns.barplot(
        data=shift,
        y="model",
        x="mean_abs_elo_points",
        hue="effect",
        order=model_order,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Magnitude of AF-quality effects")
    axes[0, 1].set_xlabel("Mean absolute Elo points")
    axes[0, 1].set_ylabel("")

    spread = summary.melt(
        id_vars="model",
        value_vars=[
            "weighted_rating_domain_spread",
            "raw_rating_domain_spread",
            "plddt_adjusted_rating_domain_spread",
        ],
        var_name="rating_type",
        value_name="domain_mean_spread",
    )
    spread["rating_type"] = spread["rating_type"].map(
        {
            "weighted_rating_domain_spread": "pLDDT-weighted",
            "raw_rating_domain_spread": "raw/unweighted",
            "plddt_adjusted_rating_domain_spread": "pLDDT-adjusted",
        }
    )
    sns.pointplot(
        data=spread,
        y="model",
        x="domain_mean_spread",
        hue="rating_type",
        order=model_order,
        dodge=0.45,
        linestyle="none",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Domain mean spread across rating definitions")
    axes[1, 0].set_xlabel("Max domain mean - min domain mean")
    axes[1, 0].set_ylabel("")

    heat = domain_summary.pivot(
        index="model", columns="domain", values="mean_plddt_control_shift"
    ).reindex(model_order)
    sns.heatmap(
        heat,
        center=0,
        cmap="coolwarm",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Mean Elo shift from pLDDT control by domain")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")

    fig.suptitle(
        "AF-structure quality effects on species Elo ratings",
        fontweight="bold",
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            out_dir / f"plddt_weighted_raw_adjusted_comparison.{suffix}",
            bbox_inches="tight",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def write_summary(
    path: Path,
    summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
) -> None:
    compact = summary[
        [
            "model",
            "n_species",
            "weighted_raw_pearson_r",
            "weighted_raw_mae",
            "raw_plddt_pearson_r",
            "raw_plddt_r_squared",
            "mean_abs_plddt_expected_component",
            "weighted_rating_domain_spread",
            "raw_rating_domain_spread",
            "plddt_adjusted_rating_domain_spread",
            "control_percent_domain_spread_remaining",
        ]
    ]
    domain_compact = domain_summary[
        [
            "model",
            "domain",
            "n_species",
            "mean_weighted_rating",
            "mean_raw_rating",
            "mean_plddt_adjusted_rating",
            "mean_weighting_shift",
            "mean_plddt_control_shift",
            "mean_plddt_elo",
            "mean_species_avg_plddt",
        ]
    ]

    lines = [
        "Weighted vs raw vs pLDDT-adjusted Elo comparison",
        "=" * 51,
        "",
        "Interpretation:",
        "  raw/unweighted Elo: model-score species Elo with every comparison weighted 1.0.",
        "  pLDDT-weighted Elo: original Elo where high-confidence AF structures count more.",
        "  pLDDT-adjusted Elo: raw Elo after subtracting the part predicted by pLDDT-only Elo.",
        "",
        "Two AF-quality effects:",
        "  weighting effect = weighted_rating - raw_rating.",
        "  baseline/confounding effect = raw_rating - pLDDT_adjusted_rating.",
        "",
        "Model-level comparison:",
        compact.to_string(index=False),
        "",
        "Domain-level comparison:",
        domain_compact.to_string(index=False),
        "",
        "Reading guide:",
        "  Large weighted_raw_mae means the choice to pLDDT-weight Elo updates changes species ranks.",
        "  Large raw_plddt_r_squared means AF confidence predicts the model's species Elo.",
        "  Large mean_abs_plddt_expected_component means many Elo points are attributable to pLDDT baseline.",
        "  If adjusted domain spread shrinks, the domain pattern was partly pLDDT-associated.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighted", type=Path, default=DEFAULT_WEIGHTED)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--adjusted", type=Path, default=DEFAULT_ADJUSTED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    results_dir = args.out_dir / "results"
    figure_dir = args.out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison = load_comparison(args.weighted, args.raw, args.adjusted)
    summary = model_summary(comparison)
    domain_summary = group_summary(comparison, "domain")
    kingdom_summary = group_summary(comparison, "lineage_kingdom")

    comparison.to_csv(results_dir / "weighted_raw_adjusted_species_comparison.csv", index=False)
    summary.to_csv(results_dir / "weighted_raw_adjusted_model_summary.csv", index=False)
    domain_summary.to_csv(results_dir / "weighted_raw_adjusted_domain_summary.csv", index=False)
    kingdom_summary.to_csv(results_dir / "weighted_raw_adjusted_kingdom_summary.csv", index=False)
    write_summary(
        results_dir / "weighted_raw_adjusted_elo_interpretation.txt",
        summary,
        domain_summary,
    )
    plot_comparison(summary, domain_summary, figure_dir)

    print(f"Wrote {results_dir / 'weighted_raw_adjusted_elo_interpretation.txt'}")
    print(f"Wrote {figure_dir / 'plddt_weighted_raw_adjusted_comparison.png'}")
    print(
        summary[
            [
                "model",
                "weighted_raw_mae",
                "raw_plddt_r_squared",
                "mean_abs_plddt_expected_component",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
