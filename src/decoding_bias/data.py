"""Loading and complete-case selection for the analysis table.

Read-and-filter helpers shared by the analysis stages: table loading, domain
filtering, and complete-case selection on the feature and score columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import catalog


def load_analysis_table(path: str | Path, *, domains_only: bool = True) -> pd.DataFrame:
    """Read the analysis table; optionally keep only the three canonical domains."""
    df = pd.read_csv(path, low_memory=False)
    if domains_only:
        df = df[df[catalog.DOMAIN_COL].isin(catalog.DOMAINS)].copy()
    return df


def complete_case(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rows with all of `columns` present, index reset (variance-decomposition style)."""
    return df.dropna(subset=list(columns)).reset_index(drop=True)


def dataset_composition(df: pd.DataFrame) -> dict:
    """Reproduce the main-dataset composition counts (paper Table 7).

    Returns proteins/species/family counts overall and per domain.
    """
    out: dict = {
        "n_proteins": int(len(df)),
        "n_species": int(df[catalog.SPECIES_COL].nunique()),
        "n_families": int(df[catalog.FAMILY_COL].nunique()),
        "by_domain": {},
    }
    for dom in catalog.DOMAINS:
        sub = df[df[catalog.DOMAIN_COL] == dom]
        out["by_domain"][dom] = {
            "proteins": int(len(sub)),
            "proteins_pct": round(100 * len(sub) / len(df), 1),
            "species": int(sub[catalog.SPECIES_COL].nunique()),
        }
    # family domain-span (three/two/single-domain families)
    span = df.groupby(catalog.FAMILY_COL)[catalog.DOMAIN_COL].nunique()
    out["families_three_domain"] = int((span == 3).sum())
    out["families_two_domain"] = int((span == 2).sum())
    out["families_single_domain"] = int((span == 1).sum())
    return out
