"""Taxonomic-preference Elo analysis (paper Fig 2, Table 3; SI Fig S1, Tables S5-S8).

The Elo algorithm lives in the reusable, seeded (deterministic) `elo_rating` module;
`elo_figures` and `elo_paper_figures` build the interactive and publication panels.
This module orchestrates the runs against a `Config`.

Arms (score-column sets, from catalog):
  full   : the 14-model cohort            (Fig 2, Table 3)
  ft020  : base 020 + AlkSec/AcidSec 020  (post-fine-tuning Elo, Fig 5)

Weighting schemes (Table S5):
  unweighted (primary) | plddt_weighted | plddt_residual

Every run writes species Elo ratings (long/wide), the per-domain summary
(model_analysis_summary.txt = Table 3 / S6 / S7 source) and the Archaea-Eukaryota
gap table. The Fig 2 publication panels (2A/2B/2C) require the species-taxonomy
metadata table (paths.metadata_table); without it the ratings, summary and gap
are still produced and only the phylum heatmap (Fig 2B) is skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import elo_rating, elo_figures, elo_paper_figures
from .. import catalog

ARMS = {"full": catalog.ELO_FULL_COLUMNS, "ft020": catalog.ELO_FT_COLUMNS}
WEIGHTINGS = {
    "unweighted": dict(use_plddt_weighting=False),
    "plddt_weighted": dict(use_plddt_weighting=True),
    "plddt_residual": dict(use_plddt_weighting=False, plddt_residual=True),
}
DOM = catalog.DOMAINS
PRIMARY = ("full", "unweighted")   # Fig 2 / Table 3


def _gap_table(long: pd.DataFrame) -> pd.DataFrame:
    """Mean species Elo by domain + Archaea-Eukaryota gap (paper Table 3)."""
    piv = (long.groupby(["model", "domain"])["rating"].mean()
               .unstack("domain").reindex(columns=DOM))
    piv["Archaea_minus_Eukaryota"] = piv["Archaea"] - piv["Eukaryota"]
    piv["top_domain"] = piv[DOM].idxmax(axis=1)
    return piv.reset_index()


def run_one(cfg, arm: str = "full", weighting: str = "unweighted",
            out_dir: Path | None = None, make_paper_figures: bool = True) -> pd.DataFrame:
    """One canonical Elo run. Returns the Archaea-Eukaryota gap table."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}; choose from {list(WEIGHTINGS)}")
    name = f"elo_{arm}_{weighting}"
    base = out_dir or cfg.stage_output("elo")
    outdir = Path(base) / name
    params = cfg.params("elo")
    models = ARMS[arm]

    elo_rating.run_full_elo_analysis(
        str(cfg.analysis_table), str(outdir), score_columns=models,
        n_permutations=params.get("n_permutations", 50),
        protein_column=params.get("protein_column", "protein_family"),
        **WEIGHTINGS[weighting],
    )
    long = pd.read_csv(outdir / "results" / "all_models_species_ratings_long.csv")
    gap = _gap_table(long)
    gap.to_csv(outdir / "results" / "archaea_eukaryota_gap.csv", index=False)
    print(f"[elo] {name}: {long['model'].nunique()} models -> {outdir}")

    # Fig 2 publication panels: only for the canonical primary run, and only if
    # the species-taxonomy metadata is available (needed for the phylum heatmap).
    if make_paper_figures and (arm, weighting) == PRIMARY:
        figdir = outdir / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        tax = None
        meta = cfg.external("metadata_table")
        if meta and meta.exists():
            tax = pd.read_csv(meta, low_memory=False)
        try:
            elo_paper_figures.build_all(
                long, tax, figdir, title=name,
                struct="proteinmpnn_score", seq="ESM2_15B_pppl_score",
                violin_points=False, gap_seed=0)
        except Exception as ex:  # pragma: no cover - presentation only
            print(f"[elo] paper-figure step skipped: {ex}")
    return gap


def run(cfg, arms=("full",), weightings=("unweighted",),
        out_dir: Path | None = None, make_paper_figures: bool = True) -> dict:
    """Run every (arm, weighting) combination requested. Returns {name: gap_df}."""
    os.environ.setdefault("WRITE_PLOTLY_PDFS", "0")   # HTML only unless kaleido present
    results = {}
    for arm in arms:
        for w in weightings:
            results[f"elo_{arm}_{w}"] = run_one(cfg, arm, w, out_dir, make_paper_figures)
    return results
