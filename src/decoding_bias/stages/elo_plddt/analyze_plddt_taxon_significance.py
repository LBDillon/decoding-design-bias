"""Test whether pLDDT differs significantly across domain/kingdom groups.

Inputs:
  dataset_update/kingdom_plddt/plddt_per_entry_by_kingdom_plot_data.csv

Outputs:
  dataset_update/kingdom_plddt/plddt_taxon_significance_summary.txt
  dataset_update/kingdom_plddt/plddt_domain_pairwise_tests.csv
  dataset_update/kingdom_plddt/plddt_kingdom_pairwise_tests.csv
  dataset_update/kingdom_plddt/plddt_omnibus_tests.csv
  dataset_update/kingdom_plddt/plddt_controlled_ols_tests.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = (
    REPO_ROOT
    / "dataset_update"
    / "kingdom_plddt"
    / "plddt_per_entry_by_kingdom_plot_data.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "dataset_update" / "kingdom_plddt"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p == 0:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def pairwise_tests(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    groups = [g for g in df[group_col].dropna().unique()]
    for a_name, b_name in combinations(groups, 2):
        a = df.loc[df[group_col] == a_name, "avg_plddt"].dropna()
        b = df.loc[df[group_col] == b_name, "avg_plddt"].dropna()
        t = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        rank_biserial = 2 * mw.statistic / (len(a) * len(b)) - 1
        rows.append(
            {
                "group_a": a_name,
                "group_b": b_name,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "mean_diff_a_minus_b": a.mean() - b.mean(),
                "median_a": a.median(),
                "median_b": b.median(),
                "median_diff_a_minus_b": a.median() - b.median(),
                "welch_t": t.statistic,
                "welch_p": t.pvalue,
                "mannwhitney_u": mw.statistic,
                "mannwhitney_p": mw.pvalue,
                "cohen_d_a_minus_b": cohen_d(a, b),
                "rank_biserial_a_minus_b": rank_biserial,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ["welch_p", "mannwhitney_p"]:
            out[f"{col}_holm"] = multipletests(out[col], method="holm")[1]
    return out


def omnibus_tests(df: pd.DataFrame, group_col: str) -> dict:
    groups = [g["avg_plddt"].dropna().to_numpy() for _, g in df.groupby(group_col, observed=True)]
    n = sum(len(g) for g in groups)
    k = len(groups)
    kw = stats.kruskal(*groups)
    anova = stats.f_oneway(*groups)

    grand_mean = df["avg_plddt"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((df["avg_plddt"] - grand_mean) ** 2).sum()
    eta_squared = ss_between / ss_total
    epsilon_squared = (kw.statistic - k + 1) / (n - k)
    return {
        "grouping": group_col,
        "n": n,
        "n_groups": k,
        "anova_f": anova.statistic,
        "anova_p": anova.pvalue,
        "eta_squared": eta_squared,
        "kruskal_h": kw.statistic,
        "kruskal_p": kw.pvalue,
        "kruskal_epsilon_squared": epsilon_squared,
    }


def controlled_ols_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Nested OLS tests controlling for length and protein-name fixed effects."""
    analysis = df.dropna(
        subset=["avg_plddt", "sequence_length", "protein_name", "domain", "lineage_kingdom"]
    ).copy()
    analysis["log_sequence_length"] = np.log(analysis["sequence_length"])

    base = smf.ols("avg_plddt ~ log_sequence_length + C(protein_name)", data=analysis).fit()
    domain = smf.ols(
        "avg_plddt ~ log_sequence_length + C(protein_name) + C(domain)", data=analysis
    ).fit()
    kingdom = smf.ols(
        "avg_plddt ~ log_sequence_length + C(protein_name) + C(lineage_kingdom)",
        data=analysis,
    ).fit()

    rows = []
    for label, model in [("domain", domain), ("lineage_kingdom", kingdom)]:
        f_stat, p_value, df_diff = model.compare_f_test(base)
        rows.append(
            {
                "added_grouping": label,
                "n": int(model.nobs),
                "base_r2_length_plus_protein_name": base.rsquared,
                "full_r2": model.rsquared,
                "additional_r2": model.rsquared - base.rsquared,
                "nested_f": f_stat,
                "nested_p": p_value,
                "df_diff": df_diff,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    df["lineage_kingdom"] = df["lineage_kingdom"].fillna("Unknown")

    omnibus = pd.DataFrame(
        [omnibus_tests(df, "domain"), omnibus_tests(df, "lineage_kingdom")]
    )
    domain_pairwise = pairwise_tests(df, "domain")
    kingdom_pairwise = pairwise_tests(df, "lineage_kingdom")
    controlled = controlled_ols_tests(df)

    omnibus_path = args.out_dir / "plddt_omnibus_tests.csv"
    domain_pairwise_path = args.out_dir / "plddt_domain_pairwise_tests.csv"
    kingdom_pairwise_path = args.out_dir / "plddt_kingdom_pairwise_tests.csv"
    controlled_path = args.out_dir / "plddt_controlled_ols_tests.csv"
    summary_path = args.out_dir / "plddt_taxon_significance_summary.txt"

    omnibus.to_csv(omnibus_path, index=False)
    domain_pairwise.to_csv(domain_pairwise_path, index=False)
    kingdom_pairwise.to_csv(kingdom_pairwise_path, index=False)
    controlled.to_csv(controlled_path, index=False)

    domain_summary = df.groupby("domain")["avg_plddt"].agg(["count", "mean", "median", "std"])
    kingdom_summary = df.groupby("lineage_kingdom")["avg_plddt"].agg(["count", "mean", "median", "std"]).sort_values("count", ascending=False)

    strongest_domain = domain_pairwise.sort_values("mannwhitney_p_holm").copy()
    strongest_domain["welch_p_holm_fmt"] = strongest_domain["welch_p_holm"].map(fmt_p)
    strongest_domain["mannwhitney_p_holm_fmt"] = strongest_domain["mannwhitney_p_holm"].map(fmt_p)

    strongest_kingdom = kingdom_pairwise.sort_values("mannwhitney_p_holm").copy()
    strongest_kingdom["welch_p_holm_fmt"] = strongest_kingdom["welch_p_holm"].map(fmt_p)
    strongest_kingdom["mannwhitney_p_holm_fmt"] = strongest_kingdom["mannwhitney_p_holm"].map(fmt_p)

    lines = [
        "pLDDT taxon significance tests",
        "=" * 31,
        "",
        "Domain summary:",
        domain_summary.to_string(),
        "",
        "Kingdom summary:",
        kingdom_summary.to_string(),
        "",
        "Omnibus tests:",
        omnibus.assign(
            anova_p_fmt=omnibus["anova_p"].map(fmt_p),
            kruskal_p_fmt=omnibus["kruskal_p"].map(fmt_p),
        )[
            [
                "grouping",
                "n",
                "n_groups",
                "anova_f",
                "anova_p_fmt",
                "eta_squared",
                "kruskal_h",
                "kruskal_p_fmt",
                "kruskal_epsilon_squared",
            ]
        ].to_string(index=False),
        "",
        "Domain pairwise tests, Holm-corrected:",
        strongest_domain[
            [
                "group_a",
                "group_b",
                "mean_diff_a_minus_b",
                "median_diff_a_minus_b",
                "cohen_d_a_minus_b",
                "rank_biserial_a_minus_b",
                "welch_p_holm_fmt",
                "mannwhitney_p_holm_fmt",
            ]
        ].to_string(index=False),
        "",
        "Strongest kingdom pairwise differences, Holm-corrected:",
        strongest_kingdom[
            [
                "group_a",
                "group_b",
                "mean_diff_a_minus_b",
                "median_diff_a_minus_b",
                "cohen_d_a_minus_b",
                "rank_biserial_a_minus_b",
                "welch_p_holm_fmt",
                "mannwhitney_p_holm_fmt",
            ]
        ].head(15).to_string(index=False),
        "",
        "Controlled nested OLS tests:",
        controlled.assign(nested_p_fmt=controlled["nested_p"].map(fmt_p))[
            [
                "added_grouping",
                "n",
                "base_r2_length_plus_protein_name",
                "full_r2",
                "additional_r2",
                "nested_f",
                "nested_p_fmt",
            ]
        ].to_string(index=False),
        "",
        "Interpretation note:",
        "  The per-entry tests treat entries as independent. With thousands of",
        "  related proteins, tiny p-values are expected. Effect sizes and the",
        "  controlled additional R2 are more useful for deciding whether the",
        "  pLDDT imbalance is practically important.",
    ]
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {omnibus_path}")
    print(f"Wrote {domain_pairwise_path}")
    print(f"Wrote {kingdom_pairwise_path}")
    print(f"Wrote {controlled_path}")
    print("\n".join(lines[:45]))


if __name__ == "__main__":
    main()
