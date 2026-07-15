"""
Elo rating analysis for protein structure prediction models.

Computes Elo ratings for species across multiple inverse-folding and language
models. By default, pLDDT is used as a structural-confidence weight; this can be
disabled for baselines where pLDDT itself is treated as the score. Scores are
first z-score normalised within each protein group so that proteins of different
lengths and difficulties are comparable.

Typical usage
-------------
From the command line::

    python -m src.analysis.elo_rating --data Decoding_Bias_Dataset.csv --output-dir outputs/elo/

Programmatically::

    from decoding_bias.analysis.elo_rating import run_full_elo_analysis
    results_long, results_wide = run_full_elo_analysis("data.csv", "outputs/elo/")
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as spstats
import seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
TOL = 1e-8  # z-score difference regarded as a tie

# Default score columns present in Decoding_Bias_Dataset.csv
DEFAULT_SCORE_COLUMNS: List[str] = [
    "proteinmpnn_score",
    "esmif_score",
    "mif_score",
    "mifst_score",
    "carp_640M_score",
    "ESM2_15B_pppl_score",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


def normalize_scores(
    data: pd.DataFrame,
    score_columns: List[str],
    protein_column: str = "protein_name",
) -> pd.DataFrame:
    """Normalise scores to within-protein z-scores.

    For each *score_column* the function groups by *protein_column* and converts
    raw values to z-scores (mean 0, std 1) within each group.  This removes
    protein-level difficulty effects so that species can be compared fairly.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing protein scores.
    score_columns : list[str]
        Column names containing raw model scores.
    protein_column : str
        Column that identifies distinct proteins.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with normalised score columns (in-place where column
        names already end with ``_score``; otherwise a ``_score`` suffix is
        appended).
    """
    normalized_data = data.copy()

    for col in score_columns:
        if col not in normalized_data.columns:
            logger.warning("Column %s not found in data, skipping normalisation", col)
            continue

        normalized_col = f"{col}_score" if not col.endswith("_score") else col

        def _zscore(x: pd.Series) -> pd.Series:
            std = x.std()
            if pd.isna(std) or std == 0:
                std = 1.0
            return (x - x.mean()) / std

        group = normalized_data.groupby(protein_column)[col]
        normalized_data[normalized_col] = group.transform(_zscore)

        logger.info("Normalised %s -> %s", col, normalized_col)
        logger.info(
            "  Original range: [%.2f, %.2f]",
            normalized_data[col].min(),
            normalized_data[col].max(),
        )
        logger.info(
            "  Normalised range: [%.2f, %.2f]",
            normalized_data[normalized_col].min(),
            normalized_data[normalized_col].max(),
        )

    return normalized_data


def residualize_on_plddt(
    data: pd.DataFrame,
    score_columns: List[str],
    plddt_column: str = "avg_plddt",
) -> pd.DataFrame:
    """Remove the linear pLDDT component from each score (R1.1 robustness).

    For every model, the score is regressed on ``plddt_column`` by ordinary least
    squares across all proteins and replaced by its residual, so structural
    confidence is removed from the *score* itself rather than from the match
    weight.  Intended to be run unweighted.
    """
    out = data.copy()
    if plddt_column not in out.columns:
        logger.warning("pLDDT column %s absent; residualisation skipped", plddt_column)
        return out
    p = pd.to_numeric(out[plddt_column], errors="coerce")
    for col in score_columns:
        c = col if col.endswith("_score") else f"{col}_score"
        if c not in out.columns:
            continue
        y = pd.to_numeric(out[c], errors="coerce")
        mask = y.notna() & p.notna()
        if mask.sum() < 10:
            continue
        b = np.polyfit(p[mask].to_numpy(), y[mask].to_numpy(), 1)
        out[c] = y - (b[0] * p + b[1])
        logger.info("Residualised %s on %s (slope %.4g, n=%d)", c, plddt_column, b[0], int(mask.sum()))
    return out


# ---------------------------------------------------------------------------
# K-factor strategies
# ---------------------------------------------------------------------------


class KStrategy:
    """Base class -- implement ``__call__(rating, matches_played) -> K``."""

    def __call__(self, rating: float, matches_played: int) -> float:
        raise NotImplementedError


class FixedK(KStrategy):
    """Constant K factor."""

    def __init__(self, K: int = 32) -> None:
        self.K = K

    def __call__(self, rating: float, matches_played: int) -> float:
        return self.K


class DecayK(KStrategy):
    """Exponentially decaying K schedule.

    Parameters
    ----------
    K0 : int
        Starting K for an unrated species.
    K_min : int
        Floor value K approaches as matches grow.
    halftime : int
        Matches needed for K to halve from *K0* toward *K_min*.
    """

    def __init__(self, K0: int = 40, K_min: int = 10, halftime: int = 400) -> None:
        self.K0 = K0
        self.K_min = K_min
        self.halftime = halftime

    def __call__(self, rating: float, matches_played: int) -> float:
        return self.K_min + (self.K0 - self.K_min) * 0.5 ** (
            matches_played / self.halftime
        )


# ---------------------------------------------------------------------------
# Elo rater
# ---------------------------------------------------------------------------


class StandardEloRater:
    """Elo engine with pluggable K strategies and quality weights.

    Parameters
    ----------
    k_strategy : KStrategy or None
        How K is determined for each update.  Defaults to ``FixedK(32)``.
    initial_rating : float
        Starting rating for every new species.
    lower, upper : float
        Hard clamps on the rating scale.
    """

    def __init__(
        self,
        k_strategy: Optional[KStrategy] = None,
        initial_rating: float = 1500.0,
        lower: float = 100.0,
        upper: float = 3000.0,
    ) -> None:
        self.k_strategy: KStrategy = k_strategy or FixedK(32)
        self.initial_rating = initial_rating
        self.lower = lower
        self.upper = upper
        self.ratings: Dict[str, float] = defaultdict(lambda: self.initial_rating)
        self.matches: Dict[str, int] = defaultdict(int)

    def update_rating(
        self,
        player_a: str,
        player_b: str,
        score_a: float,
        quality_weight: float = 1.0,
    ) -> Tuple[float, float]:
        """Update ratings for a single match.

        Parameters
        ----------
        player_a, player_b : str
            Species identifiers.
        score_a : float
            Outcome from *player_a*'s perspective (1 = win, 0.5 = tie, 0 = loss).
        quality_weight : float
            Multiplicative weight in (0, 1] applied to K.  Derived from pLDDT
            to down-weight matches involving low-confidence structures.

        Returns
        -------
        tuple[float, float]
            Updated ratings for (player_a, player_b).
        """
        r_a, r_b = self.ratings[player_a], self.ratings[player_b]
        k_a = self.k_strategy(r_a, self.matches[player_a]) * quality_weight
        k_b = self.k_strategy(r_b, self.matches[player_b]) * quality_weight

        exp_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        delta = score_a - exp_a

        r_a_new = float(np.clip(r_a + k_a * delta, self.lower, self.upper))
        r_b_new = float(np.clip(r_b - k_b * delta, self.lower, self.upper))

        self.ratings[player_a] = r_a_new
        self.ratings[player_b] = r_b_new
        self.matches[player_a] += 1
        self.matches[player_b] += 1
        return r_a_new, r_b_new


# ---------------------------------------------------------------------------
# Elo computation (multi-permutation)
# ---------------------------------------------------------------------------


def compute_elo_ratings(
    data: pd.DataFrame,
    score_column: str,
    protein_column: str = "protein_name",
    species_column: str = "species",
    plddt_column: str = "avg_plddt",
    n_permutations: int = 50,
    min_species_per_protein: int = 2,
    k_strategy: Optional[KStrategy] = None,
    seed: int = SEED,
    tol: float = TOL,
    use_plddt_weighting: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute Elo ratings for species.

    When ``use_plddt_weighting`` is true, every pairwise match is down-weighted
    by ``(pLDDT_i * pLDDT_j) / 10000`` so that low-confidence structures
    contribute less.

    Parameters
    ----------
    data : pd.DataFrame
        Normalised input data.
    score_column : str
        Column with the (normalised) model score.
    protein_column, species_column, plddt_column : str
        Identifier columns.
    n_permutations : int
        Number of random protein-order permutations (for stability).
    min_species_per_protein : int
        Proteins with fewer species are skipped.
    k_strategy : KStrategy or None
        K-factor strategy; defaults to ``FixedK(32)``.
    seed : int
        Random seed for reproducibility.
    tol : float
        Absolute z-score difference treated as a tie.
    use_plddt_weighting : bool
        If true, scale each Elo update by the paired structures' pLDDT values.
        If false, every comparison receives full Elo weight.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{species: {"rating", "std_err", "ci_lower", "ci_upper"}}``.
    """
    required = {score_column, species_column, protein_column}
    if use_plddt_weighting:
        required.add(plddt_column)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = data.dropna(subset=list(required)).copy()
    if df.empty:
        raise ValueError("No valid data after dropping NaNs.")

    ok_proteins = (
        df.groupby(protein_column)[species_column]
        .nunique()
        .loc[lambda s: s >= min_species_per_protein]
        .index
    )
    logger.info(
        "Found %d proteins with >= %d species", len(ok_proteins), min_species_per_protein
    )
    if use_plddt_weighting:
        logger.info("Using %s as the Elo update weight column", plddt_column)
    else:
        logger.info("pLDDT weighting disabled; every Elo update has weight 1.0")

    all_ratings: Dict[str, List[float]] = defaultdict(list)
    k_strategy = k_strategy or FixedK(32)

    # Precompute, once, the per-protein species medians (+ optional pLDDT weight) and
    # pairwise indices, with species mapped to integer ids. This is
    # permutation-independent and lets the hot loop use plain array indexing instead of
    # a per-protein groupby and dict updates (~9x faster; ratings are bit-identical to
    # the StandardEloRater path -- verified).
    agg_spec = {"score": (score_column, "median")}
    if use_plddt_weighting:
        agg_spec["qual"] = (plddt_column, "mean")
    ok_set = set(ok_proteins)
    sub_all = df[df[protein_column].isin(ok_set)]
    species_list = sorted(sub_all[species_column].unique())
    sp_to_int = {s: i for i, s in enumerate(species_list)}
    n_sp = len(species_list)
    prot_cache: Dict[object, tuple] = {}
    for protein, sub in sub_all.groupby(protein_column):
        agg = sub.groupby(species_column).agg(**agg_spec)
        sid = np.fromiter((sp_to_int[s] for s in agg.index), dtype=np.int64, count=len(agg))
        idx_i, idx_j = np.triu_indices(len(sid), k=1)
        prot_cache[protein] = (
            sid,
            agg["score"].to_numpy(),
            agg["qual"].to_numpy() if use_plddt_weighting else None,
            idx_i,
            idx_j,
        )
    ok_arr = np.asarray(ok_proteins)
    fixed_k = k_strategy.K if isinstance(k_strategy, FixedK) else None
    lower, upper = 100.0, 3000.0

    for perm_idx in tqdm(range(n_permutations), desc=f"Elo on {score_column}"):
        R = np.full(n_sp, 1500.0)
        M = np.zeros(n_sp, dtype=np.int64)
        for protein in np.random.default_rng(seed + perm_idx).permutation(ok_arr):
            sid, scores, quals, idx_i, idx_j = prot_cache[protein]
            for t in range(len(idx_i)):
                i = idx_i[t]; j = idx_j[t]
                a = sid[i]; b = sid[j]
                diff = scores[i] - scores[j]
                s_a = 0.5 if abs(diff) < tol else (1.0 if diff > 0.0 else 0.0)
                w = (min(1.0, max(0.01, (quals[i] * quals[j]) / 10_000.0))
                     if use_plddt_weighting else 1.0)
                ra = R[a]; rb = R[b]
                k_a = (fixed_k if fixed_k is not None else k_strategy(ra, M[a])) * w
                k_b = (fixed_k if fixed_k is not None else k_strategy(rb, M[b])) * w
                exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
                delta = s_a - exp_a
                R[a] = min(upper, max(lower, ra + k_a * delta))
                R[b] = min(upper, max(lower, rb - k_b * delta))
                M[a] += 1; M[b] += 1
        for sp, i in sp_to_int.items():
            all_ratings[sp].append(float(R[i]))

    final: Dict[str, Dict[str, float]] = {}
    for sp, arr in all_ratings.items():
        a = np.asarray(arr, dtype=float)
        final[sp] = {
            "rating": float(a.mean()),
            "std_err": float(a.std(ddof=1) / np.sqrt(len(a))),
            "ci_lower": float(np.percentile(a, 2.5)),
            "ci_upper": float(np.percentile(a, 97.5)),
        }
    return final


# ---------------------------------------------------------------------------
# Effect-size helpers
# ---------------------------------------------------------------------------


def calculate_cohens_d(
    group1: pd.Series, group2: pd.Series
) -> float:
    """Calculate Cohen's d effect size between two groups.

    Uses the pooled standard deviation as the denominator.

    Parameters
    ----------
    group1, group2 : pd.Series
        Rating arrays for two groups (e.g. two taxonomic domains).

    Returns
    -------
    float
        Cohen's d (positive means *group1* > *group2*).
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_sd)


def calculate_domain_statistics(
    df: pd.DataFrame,
) -> Dict[str, dict]:
    """Compute per-model ANOVA and pairwise Cohen's d across domains.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format results with columns ``model``, ``domain``, ``rating``.

    Returns
    -------
    dict[str, dict]
        Keyed by model name. Each value contains ``domain_stats`` (DataFrame),
        ``anova`` (dict with ``f_statistic`` and ``p_value``), and
        ``effect_sizes`` (list of dicts).
    """
    stats_by_model: Dict[str, dict] = {}

    for model in df["model"].unique():
        model_df = df[df["model"] == model]

        domain_stats = (
            model_df.groupby("domain")
            .agg(
                {
                    "rating": ["count", "mean", "std", "sem"],
                    "ci_lower": "mean",
                    "ci_upper": "mean",
                }
            )
            .round(2)
        )

        domain_groups = [
            group["rating"].values for _, group in model_df.groupby("domain")
        ]
        if len(domain_groups) >= 2 and all(len(g) > 0 for g in domain_groups):
            f_stat, p_val = spstats.f_oneway(*domain_groups)
        else:
            f_stat, p_val = float("nan"), float("nan")

        domains = model_df["domain"].unique()
        effect_sizes = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                g1 = model_df.loc[model_df["domain"] == domains[i], "rating"]
                g2 = model_df.loc[model_df["domain"] == domains[j], "rating"]
                if len(g1) > 0 and len(g2) > 0:
                    effect_sizes.append(
                        {
                            "comparison": f"{domains[i]} vs {domains[j]}",
                            "cohens_d": calculate_cohens_d(g1, g2),
                        }
                    )

        stats_by_model[model] = {
            "domain_stats": domain_stats,
            "anova": {"f_statistic": f_stat, "p_value": p_val},
            "effect_sizes": effect_sizes,
        }

    return stats_by_model


def calculate_model_correlations(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pearson correlation matrix across models' Elo ratings.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format results with ``species``, ``model``, ``rating``.

    Returns
    -------
    dict
        ``{"correlations": DataFrame, "p_values": DataFrame}``.
    """
    models = df["model"].unique()
    pivot_df = df.pivot(index="species", columns="model", values="rating")

    correlations = pivot_df.corr()
    p_values = pd.DataFrame(index=models, columns=models, dtype=float)

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            common = pivot_df[[m1, m2]].dropna()
            if len(common) >= 2:
                _, p = spstats.pearsonr(common[m1], common[m2])
                p_values.loc[m1, m2] = p
                p_values.loc[m2, m1] = p
            else:
                p_values.loc[m1, m2] = float("nan")
                p_values.loc[m2, m1] = float("nan")

    return {"correlations": correlations, "p_values": p_values}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_elo_distribution(
    df: pd.DataFrame, output_dir: str, baseline_rating: float = 1500.0
) -> str:
    """Bar chart of Elo deviations from baseline, one PNG per model."""
    os.makedirs(output_dir, exist_ok=True)

    domain_colors = {
        "Bacteria": "#3498db",
        "bacteria": "#3498db",
        "Archaea": "#e74c3c",
        "archaea": "#e74c3c",
        "Eukaryota": "#2ecc71",
        "eukarya": "#2ecc71",
    }

    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()
        model_df["rating_diff"] = model_df["rating"] - baseline_rating
        df_sorted = model_df.sort_values("rating_diff")

        plt.figure(figsize=(12, 8))
        colors = [domain_colors.get(d, "#999999") for d in df_sorted["domain"]]
        plt.bar(range(len(df_sorted)), df_sorted["rating_diff"], color=colors, alpha=0.7)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        for domain in sorted(df_sorted["domain"].unique()):
            subset = df_sorted[df_sorted["domain"] == domain]
            if not subset.empty:
                plt.axhline(
                    y=subset["rating_diff"].mean(),
                    color=domain_colors.get(domain, "#999999"),
                    linestyle="--",
                    alpha=0.8,
                    label=f"{domain.title()} Mean",
                )

        plt.title(f"{model.upper()}\nElo Rating Differences from Baseline", pad=20, fontsize=14)
        plt.ylabel("Difference from Baseline Rating (1500)", fontsize=12)
        plt.xlabel("Species", fontsize=12)
        plt.xticks([])
        plt.legend(title="Domain", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, axis="y", alpha=0.3)
        y_max = max(abs(plt.ylim()[0]), abs(plt.ylim()[1]))
        plt.ylim(-y_max, y_max)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{model}_elo_ratings_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    return output_dir


def plot_model_correlations(
    correlation_results: Dict[str, pd.DataFrame], output_dir: str
) -> str:
    """Heatmap of inter-model Elo correlation."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_results["correlations"],
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
    )
    plt.title("Model Rating Correlations", fontsize=14)
    plt.tight_layout()
    out = os.path.join(output_dir, "model_correlations_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main analysis orchestrator
# ---------------------------------------------------------------------------


def run_full_elo_analysis(
    data_path: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    *,
    protein_column: str = "protein_name",
    species_column: str = "species",
    plddt_column: str = "avg_plddt",
    n_permutations: int = 50,
    min_species_per_protein: int = 2,
    score_columns: Optional[List[str]] = None,
    use_plddt_weighting: bool = True,
    plddt_residual: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete Elo analysis pipeline.

    Parameters
    ----------
    data_path : str, Path, or pd.DataFrame
        Path to ``Decoding_Bias_Dataset.csv`` (or compatible CSV), or an
        already-loaded DataFrame.
    output_dir : str or Path
        Root directory for all outputs (CSVs, PNGs, summary text).
    protein_column : str
        Column identifying distinct proteins.
    species_column : str
        Column identifying species.
    plddt_column : str
        Column with per-residue pLDDT averages used as quality weight when
        ``use_plddt_weighting`` is true.
    n_permutations : int
        Number of protein-order permutations for Elo stability.
    min_species_per_protein : int
        Minimum species count for a protein to be included.
    score_columns : list[str] or None
        Model-score columns to analyse.  If *None*, columns ending with
        ``_score`` are auto-detected and ``DEFAULT_SCORE_COLUMNS`` is used
        as a fallback.
    use_plddt_weighting : bool
        If true, use pLDDT to down-weight low-confidence Elo comparisons.
        If false, keep the Elo updates unweighted.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(results_long, results_wide)`` -- long-format and wide-format
        DataFrames of species Elo ratings.
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"

    for d in (output_dir, results_dir, figure_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    fh = logging.FileHandler(log_dir / "elo_analysis.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if isinstance(data_path, pd.DataFrame):
        input_data = data_path.copy()
        logger.info("Using provided DataFrame (%d rows)", len(input_data))
    else:
        data_path = Path(data_path)
        logger.info("Loading data from %s", data_path)
        input_data = pd.read_csv(data_path)
        logger.info("Loaded %d rows", len(input_data))

    # ------------------------------------------------------------------
    # 2. Resolve score columns
    # ------------------------------------------------------------------
    if score_columns is None:
        # Prefer the 6 paper models when present; fall back to all *_score cols
        default_present = [
            c for c in DEFAULT_SCORE_COLUMNS if c in input_data.columns
        ]
        if default_present:
            score_columns = default_present
        else:
            score_columns = [c for c in input_data.columns if c.endswith("_score")]
        if not score_columns:
            raise ValueError(
                "No score columns provided and none auto-detected. "
                "Pass score_columns explicitly."
            )
    else:
        score_columns = [c for c in score_columns if c in input_data.columns]
        if not score_columns:
            raise ValueError(
                "None of the specified score columns were found in the data."
            )

    logger.info("Using score columns: %s", score_columns)
    logger.info("Using pLDDT weighting: %s", use_plddt_weighting)

    # ------------------------------------------------------------------
    # 3. Domain lookup
    # ------------------------------------------------------------------
    if "domain" in input_data.columns:
        species_domains = (
            input_data[[species_column, "domain"]]
            .drop_duplicates()
            .set_index(species_column)["domain"]
        )
        logger.info("Found domain info for %d species", len(species_domains))
    else:
        logger.warning("No 'domain' column found -- domain will be 'Unknown'")
        species_domains = None

    # ------------------------------------------------------------------
    # 4. Normalise scores
    # ------------------------------------------------------------------
    if plddt_residual:
        logger.info("pLDDT-residual mode: regressing each score on %s and using residuals; "
                    "pLDDT weighting disabled.", plddt_column)
        input_data = residualize_on_plddt(input_data, score_columns, plddt_column)
        use_plddt_weighting = False

    logger.info("Normalising scores...")
    normalized_data = normalize_scores(input_data.copy(), score_columns, protein_column)
    analysis_score_columns = [
        f"{col}_score" if not col.endswith("_score") else col for col in score_columns
    ]

    # ------------------------------------------------------------------
    # 5. Compute Elo ratings per model
    # ------------------------------------------------------------------
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in tqdm(analysis_score_columns, desc="Computing Elo ratings"):
        try:
            results = compute_elo_ratings(
                normalized_data,
                model,
                protein_column=protein_column,
                species_column=species_column,
                plddt_column=plddt_column,
                n_permutations=n_permutations,
                min_species_per_protein=min_species_per_protein,
                use_plddt_weighting=use_plddt_weighting,
            )
            all_results[model] = results
        except Exception:
            logger.exception("Error computing Elo ratings for %s", model)

    if not all_results:
        raise ValueError("Failed to compute Elo ratings for any model.")

    # ------------------------------------------------------------------
    # 6. Build long-format results
    # ------------------------------------------------------------------
    rows = []
    for model, model_results in all_results.items():
        for species, st in model_results.items():
            domain = (
                species_domains.get(species, "Unknown")
                if species_domains is not None
                else "Unknown"
            )
            rows.append(
                {
                    species_column: species,
                    "model": model,
                    "rating": st["rating"],
                    "std_err": st["std_err"],
                    "ci_lower": st["ci_lower"],
                    "ci_upper": st["ci_upper"],
                    "domain": domain,
                }
            )

    results_long = pd.DataFrame(rows)
    long_path = results_dir / "all_models_species_ratings_long.csv"
    results_long.to_csv(long_path, index=False)
    logger.info("Saved long-format results to %s", long_path)

    # ------------------------------------------------------------------
    # 7. Build wide-format results
    # ------------------------------------------------------------------
    df_wide = results_long.pivot(
        index=species_column,
        columns="model",
        values=["rating", "std_err", "ci_lower", "ci_upper"],
    )
    df_wide.columns = [f"{metric}_{model}" for metric, model in df_wide.columns]
    domains_col = (
        results_long.groupby(species_column)["domain"].first()
    )
    df_wide = df_wide.join(domains_col)
    df_wide.reset_index(inplace=True)
    wide_path = results_dir / "all_models_species_ratings_wide.csv"
    df_wide.to_csv(wide_path, index=False)
    logger.info("Saved wide-format results to %s", wide_path)

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    try:
        plot_elo_distribution(results_long, str(figure_dir))
    except Exception:
        logger.exception("Error creating Elo distribution plots")

    # ------------------------------------------------------------------
    # 9. Domain statistics
    # ------------------------------------------------------------------
    try:
        model_stats = calculate_domain_statistics(results_long)
    except Exception:
        logger.exception("Error calculating domain statistics")
        model_stats = {}

    # ------------------------------------------------------------------
    # 10. Model correlations
    # ------------------------------------------------------------------
    try:
        corr_results = calculate_model_correlations(results_long)
        plot_model_correlations(corr_results, str(figure_dir))
    except Exception:
        logger.exception("Error calculating model correlations")

    # ------------------------------------------------------------------
    # 11. Save summary text
    # ------------------------------------------------------------------
    stats_path = results_dir / "model_analysis_summary.txt"
    try:
        with open(stats_path, "w") as f:
            f.write("Model Analysis Summary\n")
            f.write("=====================\n\n")
            for model in sorted(model_stats.keys()):
                st = model_stats[model]
                f.write(f"\n{model} Analysis\n")
                f.write("=" * (len(model) + 9) + "\n\n")
                f.write("Domain Statistics:\n")
                f.write(str(st["domain_stats"]))
                f.write("\n\nANOVA Results:\n")
                f.write(f"F-statistic: {st['anova']['f_statistic']:.2f}\n")
                f.write(f"p-value: {st['anova']['p_value']:.2e}\n")
                f.write("\nEffect Sizes (Cohen's d):\n")
                for eff in st["effect_sizes"]:
                    f.write(f"{eff['comparison']}: {eff['cohens_d']:.2f}\n")
                f.write("\n" + "-" * 50 + "\n")
        logger.info("Saved statistical summary to %s", stats_path)
    except Exception:
        logger.exception("Error saving statistical summary")

    logger.info("Analysis complete. Results saved to %s", output_dir)
    return results_long, df_wide


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute quality-weighted Elo ratings for species across "
        "inverse-folding / language models.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the input CSV (e.g. Decoding_Bias_Dataset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where results, figures, and logs are written.",
    )
    parser.add_argument(
        "--protein-column",
        default="protein_name",
        help="Column identifying distinct proteins (default: protein_name).",
    )
    parser.add_argument(
        "--species-column",
        default="species",
        help="Column identifying species (default: species).",
    )
    parser.add_argument(
        "--plddt-column",
        default="avg_plddt",
        help="Column with average pLDDT values (default: avg_plddt).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=50,
        help="Number of protein-order permutations (default: 50).",
    )
    parser.add_argument(
        "--min-species",
        type=int,
        default=2,
        help="Minimum species per protein for inclusion (default: 2).",
    )
    parser.add_argument(
        "--score-columns",
        nargs="+",
        default=None,
        help="Score columns to analyse. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-plddt-weighting",
        action="store_false",
        dest="use_plddt_weighting",
        help="Disable pLDDT quality weighting so every Elo update has weight 1.0.",
    )
    parser.add_argument(
        "--plddt-residual",
        action="store_true",
        help="Regress each score on pLDDT and run Elo on the residuals "
        "(R1.1 robustness; implies unweighted).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    run_full_elo_analysis(
        data_path=args.data,
        output_dir=args.output_dir,
        protein_column=args.protein_column,
        species_column=args.species_column,
        plddt_column=args.plddt_column,
        n_permutations=args.n_permutations,
        min_species_per_protein=args.min_species,
        score_columns=args.score_columns,
        use_plddt_weighting=args.use_plddt_weighting,
        plddt_residual=args.plddt_residual,
    )


if __name__ == "__main__":
    main()
