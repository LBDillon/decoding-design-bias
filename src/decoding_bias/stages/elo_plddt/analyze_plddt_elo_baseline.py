"""Run an unweighted species Elo baseline with pLDDT as the score.

This answers: if pLDDT is treated like a model score, without also using pLDDT
as an Elo quality weight, which species/taxa are favored by structure
confidence alone?

Outputs are written under:
  dataset_update/kingdom_plddt/plddt_elo_baseline/
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decoding_bias.analysis.elo_rating import run_full_elo_analysis  # noqa: E402


DEFAULT_DATA = REPO_ROOT / "data" / "Decoding_Bias_Dataset.csv"
DEFAULT_KINGDOM_DATA = (
    REPO_ROOT / "dataset_update" / "kingdom_plddt" / "plddt_per_entry_by_kingdom_plot_data.csv"
)
DEFAULT_MODEL_ELO = (
    REPO_ROOT / "outputs" / "elo_analysis_10models" / "results" / "all_models_species_ratings_long.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "dataset_update" / "kingdom_plddt" / "plddt_elo_baseline"


def fmt_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value == 0:
        return "<1e-300"
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, dropna=False)
        .agg(
            n_species=("species", "size"),
            total_entries=("n_entries", "sum"),
            mean_elo=("rating", "mean"),
            median_elo=("rating", "median"),
            sd_elo=("rating", "std"),
            min_elo=("rating", "min"),
            max_elo=("rating", "max"),
            mean_species_avg_plddt=("mean_entry_plddt", "mean"),
            median_species_avg_plddt=("median_entry_plddt", "median"),
        )
        .reset_index()
        .sort_values("mean_elo", ascending=False)
    )


def pairwise_rating_tests(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    groups = [g for g in df[group_col].dropna().unique()]
    for a_name, b_name in combinations(groups, 2):
        a = df.loc[df[group_col] == a_name, "rating"].dropna()
        b = df.loc[df[group_col] == b_name, "rating"].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t_test = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        rank_biserial = 2 * mw.statistic / (len(a) * len(b)) - 1
        rows.append(
            {
                "group_a": a_name,
                "group_b": b_name,
                "n_a": len(a),
                "n_b": len(b),
                "mean_elo_a": a.mean(),
                "mean_elo_b": b.mean(),
                "mean_elo_diff_a_minus_b": a.mean() - b.mean(),
                "median_elo_a": a.median(),
                "median_elo_b": b.median(),
                "median_elo_diff_a_minus_b": a.median() - b.median(),
                "welch_t": t_test.statistic,
                "welch_p": t_test.pvalue,
                "mannwhitney_u": mw.statistic,
                "mannwhitney_p": mw.pvalue,
                "cohen_d_a_minus_b": cohen_d(a, b),
                "rank_biserial_a_minus_b": rank_biserial,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        for p_col in ["welch_p", "mannwhitney_p"]:
            out[f"{p_col}_holm"] = multipletests(out[p_col], method="holm")[1]
    return out


def model_correlations(
    plddt_elo: pd.DataFrame, model_elo_path: Path
) -> pd.DataFrame:
    if not model_elo_path.exists():
        return pd.DataFrame()

    model_elo = pd.read_csv(model_elo_path)
    baseline = plddt_elo[["species", "rating"]].rename(columns={"rating": "plddt_elo"})
    rows = []
    for model, group in model_elo.groupby("model"):
        merged = baseline.merge(
            group[["species", "rating"]].rename(columns={"rating": "model_elo"}),
            on="species",
            how="inner",
        )
        if len(merged) < 3:
            continue
        pearson = stats.pearsonr(merged["plddt_elo"], merged["model_elo"])
        spearman = stats.spearmanr(merged["plddt_elo"], merged["model_elo"])
        rows.append(
            {
                "model": model,
                "n_species": len(merged),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def plot_baseline(
    df: pd.DataFrame,
    correlations: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    domain_order = (
        df.groupby("domain")["rating"].mean().sort_values(ascending=False).index.tolist()
    )
    kingdom_order = (
        df.groupby("lineage_kingdom")["rating"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.boxplot(data=df, x="domain", y="rating", order=domain_order, ax=axes[0, 0])
    sns.stripplot(
        data=df,
        x="domain",
        y="rating",
        order=domain_order,
        color="black",
        alpha=0.45,
        size=3,
        ax=axes[0, 0],
    )
    axes[0, 0].axhline(1500, color="#555555", linestyle="--", linewidth=1)
    axes[0, 0].set_title("pLDDT-only species Elo by domain")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Elo rating")

    sns.boxplot(
        data=df,
        x="lineage_kingdom",
        y="rating",
        order=kingdom_order,
        ax=axes[0, 1],
    )
    sns.stripplot(
        data=df,
        x="lineage_kingdom",
        y="rating",
        order=kingdom_order,
        color="black",
        alpha=0.45,
        size=3,
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(1500, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_title("pLDDT-only species Elo by lineage kingdom")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Elo rating")
    axes[0, 1].tick_params(axis="x", rotation=35)

    sns.scatterplot(
        data=df,
        x="median_entry_plddt",
        y="rating",
        hue="domain",
        s=45,
        alpha=0.85,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(1500, color="#555555", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Species median pLDDT vs pLDDT-only Elo")
    axes[1, 0].set_xlabel("Species median entry pLDDT")
    axes[1, 0].set_ylabel("Elo rating")

    if not correlations.empty:
        sns.barplot(
            data=correlations,
            y="model",
            x="pearson_r",
            color="#4C78A8",
            ax=axes[1, 1],
        )
        axes[1, 1].axvline(0, color="#555555", linewidth=1)
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].set_title("Correlation with existing model Elo ratings")
        axes[1, 1].set_xlabel("Pearson r vs pLDDT-only Elo")
        axes[1, 1].set_ylabel("")
    else:
        axes[1, 1].axis("off")

    fig.suptitle(
        "Unweighted species Elo baseline using pLDDT as the score",
        fontweight="bold",
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"plddt_elo_baseline_grid.{suffix}",
            bbox_inches="tight",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def write_summary(
    out_path: Path,
    df: pd.DataFrame,
    domain_summary: pd.DataFrame,
    kingdom_summary: pd.DataFrame,
    domain_tests: pd.DataFrame,
    kingdom_tests: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    top_species = df.sort_values("rating", ascending=False).head(10)
    bottom_species = df.sort_values("rating", ascending=True).head(10)
    corr_species = df[["rating", "mean_entry_plddt", "median_entry_plddt", "n_entries"]].corr()

    lines = [
        "pLDDT-only species Elo baseline",
        "=" * 32,
        "",
        "Method:",
        "  Score column: avg_plddt, normalized within protein_name.",
        "  Elo update weighting: disabled, so every pairwise comparison has weight 1.0.",
        "  Interpretation: rating > 1500 means that species tends to have higher",
        "  pLDDT than other species for the same protein groups.",
        "",
        f"Species included: {len(df)}",
        f"Rating range: {df['rating'].min():.1f} to {df['rating'].max():.1f}",
        "",
        "Domain summary:",
        domain_summary.to_string(index=False),
        "",
        "Kingdom summary:",
        kingdom_summary.to_string(index=False),
        "",
        "Domain pairwise rating tests, Holm-corrected:",
        domain_tests.assign(
            welch_p_holm_fmt=domain_tests["welch_p_holm"].map(fmt_p),
            mannwhitney_p_holm_fmt=domain_tests["mannwhitney_p_holm"].map(fmt_p),
        )[
            [
                "group_a",
                "group_b",
                "mean_elo_diff_a_minus_b",
                "cohen_d_a_minus_b",
                "welch_p_holm_fmt",
                "mannwhitney_p_holm_fmt",
            ]
        ].to_string(index=False),
        "",
        "Strongest kingdom pairwise rating tests, Holm-corrected:",
        kingdom_tests.assign(
            welch_p_holm_fmt=kingdom_tests["welch_p_holm"].map(fmt_p),
            mannwhitney_p_holm_fmt=kingdom_tests["mannwhitney_p_holm"].map(fmt_p),
        )
        .sort_values("mannwhitney_p_holm")
        .head(15)[
            [
                "group_a",
                "group_b",
                "mean_elo_diff_a_minus_b",
                "cohen_d_a_minus_b",
                "welch_p_holm_fmt",
                "mannwhitney_p_holm_fmt",
            ]
        ]
        .to_string(index=False),
        "",
        "Correlation of pLDDT-only Elo with species-level pLDDT summaries:",
        corr_species.to_string(),
        "",
        "Correlation of pLDDT-only Elo with existing model Elo ratings:",
        correlations.assign(
            pearson_p_fmt=correlations["pearson_p"].map(fmt_p),
            spearman_p_fmt=correlations["spearman_p"].map(fmt_p),
        )[
            [
                "model",
                "n_species",
                "pearson_r",
                "pearson_p_fmt",
                "spearman_rho",
                "spearman_p_fmt",
            ]
        ].to_string(index=False)
        if not correlations.empty
        else "No existing model Elo file found.",
        "",
        "Top pLDDT-only Elo species:",
        top_species[
            ["species", "domain", "lineage_kingdom", "rating", "median_entry_plddt", "n_entries"]
        ].to_string(index=False),
        "",
        "Bottom pLDDT-only Elo species:",
        bottom_species[
            ["species", "domain", "lineage_kingdom", "rating", "median_entry_plddt", "n_entries"]
        ].to_string(index=False),
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--kingdom-data", type=Path, default=DEFAULT_KINGDOM_DATA)
    parser.add_argument("--model-elo", type=Path, default=DEFAULT_MODEL_ELO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--min-species", type=int, default=2)
    args = parser.parse_args()

    results_long, _ = run_full_elo_analysis(
        data_path=args.data,
        output_dir=args.out_dir,
        score_columns=["avg_plddt"],
        plddt_column="avg_plddt",
        n_permutations=args.n_permutations,
        min_species_per_protein=args.min_species,
        use_plddt_weighting=False,
    )

    kingdom_data = pd.read_csv(args.kingdom_data)
    species_context = (
        kingdom_data.groupby("species")
        .agg(
            lineage_kingdom=("lineage_kingdom", "first"),
            n_entries=("Entry", "size"),
            mean_entry_plddt=("avg_plddt", "mean"),
            median_entry_plddt=("avg_plddt", "median"),
        )
        .reset_index()
    )
    species_context["lineage_kingdom"] = species_context["lineage_kingdom"].fillna(
        "Unknown"
    )

    plddt_elo = results_long.merge(species_context, on="species", how="left")
    plddt_elo["lineage_kingdom"] = plddt_elo["lineage_kingdom"].fillna("Unknown")

    results_dir = args.out_dir / "results"
    figure_dir = args.out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    domain_summary = group_summary(plddt_elo, "domain")
    kingdom_summary = group_summary(plddt_elo, "lineage_kingdom")
    domain_tests = pairwise_rating_tests(plddt_elo, "domain")
    kingdom_tests = pairwise_rating_tests(plddt_elo, "lineage_kingdom")
    correlations = model_correlations(plddt_elo, args.model_elo)

    plddt_elo.to_csv(results_dir / "plddt_elo_species_context.csv", index=False)
    domain_summary.to_csv(results_dir / "plddt_elo_domain_summary.csv", index=False)
    kingdom_summary.to_csv(results_dir / "plddt_elo_kingdom_summary.csv", index=False)
    domain_tests.to_csv(results_dir / "plddt_elo_domain_pairwise_tests.csv", index=False)
    kingdom_tests.to_csv(results_dir / "plddt_elo_kingdom_pairwise_tests.csv", index=False)
    correlations.to_csv(results_dir / "plddt_elo_model_correlations.csv", index=False)

    plot_baseline(plddt_elo, correlations, figure_dir)
    write_summary(
        results_dir / "plddt_elo_baseline_summary.txt",
        plddt_elo,
        domain_summary,
        kingdom_summary,
        domain_tests,
        kingdom_tests,
        correlations,
    )

    print(f"Wrote {results_dir / 'plddt_elo_baseline_summary.txt'}")
    print(f"Wrote {figure_dir / 'plddt_elo_baseline_grid.png'}")
    print(domain_summary.to_string(index=False))
    if not correlations.empty:
        print(correlations.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
