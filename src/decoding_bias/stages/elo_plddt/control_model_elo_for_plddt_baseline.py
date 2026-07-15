"""Residualize model species Elo ratings against a pLDDT-only Elo baseline.

The recommended use is:
  1. Run model-score Elo without pLDDT quality weights.
  2. Run pLDDT-only Elo without pLDDT quality weights.
  3. For each model, fit: model_elo ~ plddt_only_elo.
  4. Use 1500 + residual as the pLDDT-adjusted model Elo.

This controls the species-level ratings for the part that is predictable from
structure confidence alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ELO = (
    REPO_ROOT
    / "dataset_update"
    / "kingdom_plddt"
    / "model_elo_unweighted"
    / "results"
    / "all_models_species_ratings_long.csv"
)
DEFAULT_PLDDT_ELO = (
    REPO_ROOT
    / "dataset_update"
    / "kingdom_plddt"
    / "plddt_elo_baseline"
    / "results"
    / "plddt_elo_species_context.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "dataset_update" / "kingdom_plddt" / "plddt_controlled_model_elo"
)


def fmt_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value == 0:
        return "<1e-300"
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def adjust_model_elo(
    model_elo: pd.DataFrame,
    plddt_elo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = plddt_elo[
        [
            "species",
            "rating",
            "lineage_kingdom",
            "n_entries",
            "mean_entry_plddt",
            "median_entry_plddt",
        ]
    ].rename(columns={"rating": "plddt_elo"})

    rows = []
    summaries = []
    for model, group in model_elo.groupby("model"):
        merged = group.merge(baseline, on="species", how="inner")
        if len(merged) < 3:
            continue

        fit = stats.linregress(merged["plddt_elo"], merged["rating"])
        expected = fit.intercept + fit.slope * merged["plddt_elo"]
        residual = merged["rating"] - expected

        merged = merged.copy()
        merged["plddt_expected_rating"] = expected
        merged["plddt_residual"] = residual
        merged["plddt_adjusted_rating"] = 1500.0 + residual
        rows.append(merged)

        summaries.append(
            {
                "model": model,
                "n_species": len(merged),
                "plddt_slope": fit.slope,
                "plddt_intercept": fit.intercept,
                "pearson_r": fit.rvalue,
                "r_squared": fit.rvalue**2,
                "p_value": fit.pvalue,
                "stderr": fit.stderr,
                "mean_raw_rating": merged["rating"].mean(),
                "mean_adjusted_rating": merged["plddt_adjusted_rating"].mean(),
                "sd_raw_rating": merged["rating"].std(),
                "sd_adjusted_rating": merged["plddt_adjusted_rating"].std(),
            }
        )

    adjusted = pd.concat(rows, ignore_index=True)
    model_summary = pd.DataFrame(summaries).sort_values("r_squared", ascending=False)
    return adjusted, model_summary


def grouped_summary(adjusted: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        adjusted.groupby(["model", group_col], dropna=False)
        .agg(
            n_species=("species", "size"),
            mean_raw_rating=("rating", "mean"),
            median_raw_rating=("rating", "median"),
            mean_expected_from_plddt=("plddt_expected_rating", "mean"),
            mean_plddt_adjusted_rating=("plddt_adjusted_rating", "mean"),
            median_plddt_adjusted_rating=("plddt_adjusted_rating", "median"),
            sd_plddt_adjusted_rating=("plddt_adjusted_rating", "std"),
            mean_plddt_elo=("plddt_elo", "mean"),
            mean_species_avg_plddt=("mean_entry_plddt", "mean"),
        )
        .reset_index()
        .sort_values(["model", "mean_plddt_adjusted_rating"], ascending=[True, False])
    )


def domain_spread(domain_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in domain_summary.groupby("model"):
        raw_spread = group["mean_raw_rating"].max() - group["mean_raw_rating"].min()
        adjusted_spread = (
            group["mean_plddt_adjusted_rating"].max()
            - group["mean_plddt_adjusted_rating"].min()
        )
        rows.append(
            {
                "model": model,
                "raw_domain_mean_spread": raw_spread,
                "plddt_adjusted_domain_mean_spread": adjusted_spread,
                "spread_change_adjusted_minus_raw": adjusted_spread - raw_spread,
                "percent_spread_remaining": adjusted_spread / raw_spread
                if raw_spread
                else pd.NA,
            }
        )
    return pd.DataFrame(rows).sort_values("raw_domain_mean_spread", ascending=False)


def plot_control_summary(
    model_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    spread: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    model_order = model_summary["model"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    sns.barplot(
        data=model_summary,
        y="model",
        x="r_squared",
        order=model_order,
        color="#4C78A8",
        ax=axes[0, 0],
    )
    axes[0, 0].set_xlim(0, max(0.05, model_summary["r_squared"].max() * 1.1))
    axes[0, 0].set_title("Species Elo variance explained by pLDDT-only Elo")
    axes[0, 0].set_xlabel("R-squared")
    axes[0, 0].set_ylabel("")

    spread_long = spread.melt(
        id_vars="model",
        value_vars=["raw_domain_mean_spread", "plddt_adjusted_domain_mean_spread"],
        var_name="rating_type",
        value_name="domain_mean_spread",
    )
    spread_long["rating_type"] = spread_long["rating_type"].map(
        {
            "raw_domain_mean_spread": "raw",
            "plddt_adjusted_domain_mean_spread": "pLDDT-adjusted",
        }
    )
    sns.pointplot(
        data=spread_long,
        y="model",
        x="domain_mean_spread",
        hue="rating_type",
        order=model_order,
        linestyle="none",
        dodge=0.35,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Domain mean spread before and after control")
    axes[0, 1].set_xlabel("Max domain mean - min domain mean")
    axes[0, 1].set_ylabel("")

    heat = domain_summary.pivot(
        index="model", columns="domain", values="mean_plddt_adjusted_rating"
    ).reindex(model_order)
    heat = heat - 1500.0
    sns.heatmap(
        heat,
        center=0,
        cmap="coolwarm",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Adjusted domain mean Elo deviation from 1500")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")

    shift = domain_summary.copy()
    shift["domain_mean_shift_after_control"] = (
        shift["mean_plddt_adjusted_rating"] - shift["mean_raw_rating"]
    )
    sns.barplot(
        data=shift,
        x="domain",
        y="domain_mean_shift_after_control",
        hue="domain",
        estimator="mean",
        errorbar=("sd", 1),
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(0, color="#555555", linewidth=1)
    axes[1, 1].set_title("Average domain mean shift after pLDDT control")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Adjusted mean - raw mean")
    if axes[1, 1].legend_ is not None:
        axes[1, 1].legend_.remove()

    fig.suptitle("Controlling model species Elo ratings for pLDDT-only Elo", fontweight="bold")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            out_dir / f"plddt_controlled_model_elo_overview.{suffix}",
            bbox_inches="tight",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def write_summary(
    path: Path,
    model_summary: pd.DataFrame,
    domain_summary: pd.DataFrame,
    kingdom_summary: pd.DataFrame,
    spread: pd.DataFrame,
) -> None:
    pivot_adjusted = (
        domain_summary.pivot(
            index="model", columns="domain", values="mean_plddt_adjusted_rating"
        )
        .reindex(model_summary["model"])
        .round(1)
    )
    kingdom_top = (
        kingdom_summary.sort_values(
            ["model", "mean_plddt_adjusted_rating"], ascending=[True, False]
        )
        .groupby("model")
        .head(3)
    )

    lines = [
        "pLDDT-controlled model species Elo",
        "=" * 35,
        "",
        "Method:",
        "  The model Elo ratings used here were rerun without pLDDT quality weights.",
        "  For each model, I fit: model_species_elo ~ plddt_only_species_elo.",
        "  pLDDT-adjusted Elo = 1500 + residual from that fit.",
        "  Positive adjusted values mean the species is better than expected after",
        "  accounting for its pLDDT-only Elo baseline.",
        "",
        "How much each model's species Elo is explained by pLDDT-only Elo:",
        model_summary.assign(p_value_fmt=model_summary["p_value"].map(fmt_p))[
            [
                "model",
                "n_species",
                "pearson_r",
                "r_squared",
                "plddt_slope",
                "p_value_fmt",
                "sd_raw_rating",
                "sd_adjusted_rating",
            ]
        ].to_string(index=False),
        "",
        "Domain mean spread before/after pLDDT control:",
        spread.to_string(index=False),
        "",
        "Adjusted domain mean Elo ratings:",
        pivot_adjusted.to_string(),
        "",
        "Top adjusted kingdoms per model:",
        kingdom_top[
            [
                "model",
                "lineage_kingdom",
                "n_species",
                "mean_raw_rating",
                "mean_plddt_adjusted_rating",
                "mean_expected_from_plddt",
            ]
        ].to_string(index=False),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-elo", type=Path, default=DEFAULT_MODEL_ELO)
    parser.add_argument("--plddt-elo", type=Path, default=DEFAULT_PLDDT_ELO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    results_dir = args.out_dir / "results"
    figure_dir = args.out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_elo = pd.read_csv(args.model_elo)
    plddt_elo = pd.read_csv(args.plddt_elo)

    adjusted, model_summary = adjust_model_elo(model_elo, plddt_elo)
    domain_summary = grouped_summary(adjusted, "domain")
    kingdom_summary = grouped_summary(adjusted, "lineage_kingdom")
    spread = domain_spread(domain_summary)

    adjusted.to_csv(results_dir / "plddt_adjusted_species_ratings_long.csv", index=False)
    model_summary.to_csv(results_dir / "plddt_adjustment_model_summary.csv", index=False)
    domain_summary.to_csv(results_dir / "plddt_adjusted_domain_summary.csv", index=False)
    kingdom_summary.to_csv(results_dir / "plddt_adjusted_kingdom_summary.csv", index=False)
    spread.to_csv(results_dir / "plddt_adjusted_domain_spread.csv", index=False)

    plot_control_summary(model_summary, domain_summary, spread, figure_dir)
    write_summary(
        results_dir / "plddt_controlled_model_elo_summary.txt",
        model_summary,
        domain_summary,
        kingdom_summary,
        spread,
    )

    print(f"Wrote {results_dir / 'plddt_controlled_model_elo_summary.txt'}")
    print(f"Wrote {figure_dir / 'plddt_controlled_model_elo_overview.png'}")
    print(model_summary[["model", "n_species", "pearson_r", "r_squared"]].to_string(index=False))


if __name__ == "__main__":
    main()
