"""Loading, filtering and composition of the main analysis table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import catalog


def load_analysis_table(path: str | Path, *, domains_only: bool = True) -> pd.DataFrame:
    """Read the table without the two excluded accessions; optionally keep only the three domains."""
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[~df[catalog.ID_COL].isin(catalog.PRIMARY_EXCLUDED_ACCESSIONS)].copy()
    if domains_only:
        df = df[df[catalog.DOMAIN_COL].isin(catalog.DOMAINS)].copy()
    return df


def dataset_composition(df: pd.DataFrame) -> dict:
    """Protein, species and family counts overall and per domain (Table S28)."""
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
    span = df.groupby(catalog.FAMILY_COL)[catalog.DOMAIN_COL].nunique()
    out["families_three_domain"] = int((span == 3).sum())
    out["families_two_domain"] = int((span == 2).sum())
    out["families_single_domain"] = int((span == 1).sum())
    return out
