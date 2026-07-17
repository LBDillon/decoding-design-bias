"""Seeded species-level Elo analysis for Figure 2 and Table 3.

Within each protein family, scores are z-normalised.  Every pair of species in a
family plays a match, families are visited in 50 seeded random orders, and the final
species ratings are averaged across those orders.  This is the complete primary Elo
calculation without the historical plotting and command-line layers.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .. import catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SEED = 42
BASELINE = 1500.0
K_FACTOR = 32.0
TOL = 1e-8


def _normalise_within_family(data: pd.DataFrame, score: str) -> pd.Series:
    def zscore(group: pd.Series) -> pd.Series:
        std = group.std()
        return (group - group.mean()) / (std if pd.notna(std) and std else 1.0)
    return data.groupby(catalog.FAMILY_COL)[score].transform(zscore)


def compute_model(
    data: pd.DataFrame,
    score: str,
    *,
    n_permutations: int = 50,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    """Compute one model's unweighted species ratings."""
    work = data[[catalog.FAMILY_COL, catalog.SPECIES_COL, score]].dropna().copy()
    work[score] = _normalise_within_family(work, score)
    valid_families = (
        work.groupby(catalog.FAMILY_COL)[catalog.SPECIES_COL].nunique()
        .loc[lambda values: values >= 2].index
    )
    work = work[work[catalog.FAMILY_COL].isin(valid_families)]
    species = sorted(work[catalog.SPECIES_COL].unique())
    species_index = {name: index for index, name in enumerate(species)}

    cache: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for family, group in work.groupby(catalog.FAMILY_COL):
        medians = group.groupby(catalog.SPECIES_COL)[score].median()
        ids = np.fromiter((species_index[name] for name in medians.index), dtype=np.int64)
        left, right = np.triu_indices(len(ids), k=1)
        cache[family] = (ids, medians.to_numpy(float), left, right)

    family_array = np.asarray(valid_families)
    runs: dict[str, list[float]] = defaultdict(list)
    for permutation in range(n_permutations):
        ratings = np.full(len(species), BASELINE)
        for family in np.random.default_rng(seed + permutation).permutation(family_array):
            ids, scores, left, right = cache[family]
            for pair in range(len(left)):
                i, j = left[pair], right[pair]
                a, b = ids[i], ids[j]
                difference = scores[i] - scores[j]
                outcome = .5 if abs(difference) < TOL else (1.0 if difference > 0 else 0.0)
                rating_a, rating_b = ratings[a], ratings[b]
                expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
                delta = outcome - expected_a
                ratings[a] = min(3000.0, max(100.0, rating_a + K_FACTOR * delta))
                ratings[b] = min(3000.0, max(100.0, rating_b - K_FACTOR * delta))
        for name, index in species_index.items():
            runs[name].append(float(ratings[index]))

    result = {}
    for name in species:
        values = np.asarray(runs[name], dtype=float)
        result[name] = {
            "rating": float(values.mean()),
            "std_err": float(values.std(ddof=1) / np.sqrt(len(values))),
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
        }
    return result


def _domain_summary(long: pd.DataFrame) -> pd.DataFrame:
    summary = long.groupby(["model", "domain"], observed=True)["rating"].mean().unstack("domain")
    summary = summary.reindex(columns=catalog.DOMAINS)
    summary["Archaea_minus_Eukaryota"] = summary["Archaea"] - summary["Eukaryota"]
    summary["top_domain"] = summary[catalog.DOMAINS].idxmax(axis=1)
    return summary.reset_index()


def _figure(long: pd.DataFrame, taxonomy: pd.DataFrame, path: Path) -> None:
    """Compact static version of the three Figure 2 panels."""
    summary = _domain_summary(long).set_index("model")
    order = [column for column in catalog.ELO_FULL_COLUMNS if column in summary.index]
    pretty = [catalog.PRETTY.get(column, column) for column in order]
    score_types = {score: kind for score, kind in catalog.FULL_COHORT.values()}
    colors = [catalog.TYPE_COLORS[score_types[column]] for column in order]

    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(2, 2, height_ratios=[.85, 1.15], hspace=.35, wspace=.28)
    ax_gap = fig.add_subplot(grid[0, 0])
    values = summary.loc[order, "Archaea_minus_Eukaryota"].to_numpy()
    y = np.arange(len(order))
    ax_gap.axvline(0, color="grey", lw=.8, ls="--")
    ax_gap.scatter(values, y, c=colors, s=38)
    ax_gap.set_yticks(y, pretty); ax_gap.invert_yaxis()
    ax_gap.set_xlabel("Archaea − Eukaryota Elo")
    ax_gap.set_title("A  Taxonomic preference gap", loc="left", weight="bold")

    ax_species = fig.add_subplot(grid[0, 1])
    for model, color, offset in [("proteinmpnn_score", "#2C7BB6", -.16),
                                 ("protgpt2_score", "#1A9641", .16)]:
        subset = long[long["model"] == model].copy()
        domain_order = {name: index for index, name in enumerate(catalog.DOMAINS)}
        subset["_domain"] = subset["domain"].map(domain_order)
        subset = subset.sort_values(["_domain", "rating"]).reset_index(drop=True)
        ax_species.plot(np.arange(len(subset)), subset["rating"] - BASELINE + offset,
                        lw=.8, label=catalog.PRETTY.get(model, model), color=color)
    ax_species.axhline(0, color="grey", lw=.8, ls="--")
    ax_species.set_xlabel("species grouped by domain"); ax_species.set_ylabel("Elo − 1500")
    ax_species.legend(frameon=False)
    ax_species.set_title("B  Representative species ratings", loc="left", weight="bold")

    ax_heat = fig.add_subplot(grid[1, :])
    tax = taxonomy[["species", "phylum_division"]].drop_duplicates("species")
    heat = long.merge(tax, on="species", how="left")
    heat["phylum_division"] = heat["phylum_division"].fillna("Other")
    heat["z"] = heat.groupby("model")["rating"].transform(lambda x: (x - x.mean()) / x.std())
    matrix = heat.groupby(["phylum_division", "model"], observed=True)["z"].mean().unstack("model")
    matrix = matrix.reindex(columns=order).dropna(how="all")
    matrix = matrix.loc[matrix.abs().mean(axis=1).sort_values(ascending=False).head(30).index]
    image = ax_heat.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax_heat.set_xticks(range(len(order)), pretty, rotation=40, ha="right")
    ax_heat.set_yticks(range(len(matrix)), matrix.index)
    ax_heat.set_title("C  Mean phylum preference (within-model z score)", loc="left", weight="bold")
    fig.colorbar(image, ax=ax_heat, label="mean species Elo z score", shrink=.7)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    cfg,
    out_dir: Path | None = None,
    *,
    n_permutations: int = 50,
    make_figure: bool = True,
) -> dict[str, pd.DataFrame]:
    out = Path(out_dir) if out_dir else cfg.stage_output("taxonomy")
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(cfg.analysis_table, low_memory=False)
    domain = data[[catalog.SPECIES_COL, catalog.DOMAIN_COL]].drop_duplicates().set_index(catalog.SPECIES_COL)[catalog.DOMAIN_COL]
    rows = []
    for index, model in enumerate(catalog.ELO_FULL_COLUMNS, start=1):
        print(f"[taxonomy] model {index}/{len(catalog.ELO_FULL_COLUMNS)}: {catalog.PRETTY.get(model, model)}")
        ratings = compute_model(data, model, n_permutations=n_permutations)
        for species, values in ratings.items():
            rows.append({"species": species, "model": model, **values, "domain": domain.get(species, "Unknown")})
    long = pd.DataFrame(rows)
    summary = _domain_summary(long)
    long.to_csv(out / "all_models_species_ratings_long.csv", index=False)
    summary.to_csv(out / "domain_elo_summary.csv", index=False)
    if make_figure:
        _figure(long, pd.read_csv(cfg.taxonomy_table), out / "figure2_taxonomy.png")
    print(f"[taxonomy] {long['species'].nunique()} species × {long['model'].nunique()} models -> {out}")
    return {"ratings": long, "summary": summary}
