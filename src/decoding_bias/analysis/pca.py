"""Biophysical PCA tables (paper Fig 3A/B; SI Table S14 loadings, compactness).

The Fig 3C / S8 GAM *landscapes* and the Table S15 GAM deviance-explained are
produced by the R/mgcv notebook (04_pca_gam/PCA_paper_figures.ipynb +
pca_corrections.R) because mgcv has no faithful Python equivalent; this module
covers the Python-computable pieces:

  * the 14-feature PCA (same SVD of standardised features as the variance
    decomposition), its loadings (Table S14) and explained-variance ratios
    (PC1 23.4%, PC2 16.7%),
  * per-domain PC1/PC2 means, 1-SD ellipse areas, and pairwise Bhattacharyya
    overlaps (the compactness table).

PC signs are oriented to the paper's convention (Table S14) so the loadings and
domain means carry the reported signs; PCA sign is otherwise arbitrary.
Deterministic (no RNG).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import catalog
from ..data import load_analysis_table

BIOPHYS = catalog.BIOPHYS_14

# Reference sign convention (paper Table S14): sequence_length loads + on PC1;
# gravy loads - on PC2. We flip each PC to match these so signs are reproducible.
_SIGN_REF = {"pc1_feature": ("sequence_length", +1.0), "pc2_feature": ("gravy", -1.0)}


def fit(df: pd.DataFrame):
    """14-feature PCA via SVD of standardised (ddof=0) features over complete rows.

    Returns (coords_df[Entry,PC1,PC2,domain], loadings_df[feature,PC1,PC2],
    explained_ratio[2])."""
    cc = df[df[BIOPHYS].notna().all(axis=1)].copy()
    Z = ((cc[BIOPHYS] - cc[BIOPHYS].mean()) / cc[BIOPHYS].std(ddof=0)).values
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    load = Vt[:2].T.copy()                       # (14, 2) principal axes
    scores = U[:, :2] * S[:2]
    # orient signs to the paper convention
    f1, s1 = _SIGN_REF["pc1_feature"]; i1 = BIOPHYS.index(f1)
    if np.sign(load[i1, 0]) != np.sign(s1):
        load[:, 0] *= -1; scores[:, 0] *= -1
    f2, s2 = _SIGN_REF["pc2_feature"]; i2 = BIOPHYS.index(f2)
    if np.sign(load[i2, 1]) != np.sign(s2):
        load[:, 1] *= -1; scores[:, 1] *= -1
    explained = (S ** 2 / (S ** 2).sum())[:2]
    coords = pd.DataFrame({"Entry": cc["Entry"].values,
                           "PC1": scores[:, 0], "PC2": scores[:, 1],
                           "domain": cc[catalog.DOMAIN_COL].values})
    loadings = pd.DataFrame({"feature": BIOPHYS, "PC1": load[:, 0], "PC2": load[:, 1]})
    return coords, loadings, explained


def _bhattacharyya(mu1, cov1, mu2, cov2) -> float:
    cov = (cov1 + cov2) / 2
    diff = (mu1 - mu2).reshape(-1, 1)
    d = 0.125 * float(diff.T @ np.linalg.inv(cov) @ diff)
    d += 0.5 * np.log(np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    return float(np.exp(-d))                      # coefficient in [0,1]


def compactness(coords: pd.DataFrame) -> pd.DataFrame:
    """Per-domain PC1/PC2 means, 1-SD ellipse area, and pairwise Bhattacharyya overlap."""
    stats = {}
    rows = []
    for dom in catalog.DOMAINS:
        sub = coords[coords.domain == dom][["PC1", "PC2"]].values
        mu = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False)
        area = float(np.pi * np.sqrt(np.linalg.det(cov)))   # 1-SD ellipse area
        stats[dom] = (mu, cov)
        rows.append(dict(domain=dom, n=len(sub), PC1_mean=mu[0], PC2_mean=mu[1],
                         ellipse_area_1sd=area))
    df = pd.DataFrame(rows)
    pairs = [("Bacteria", "Archaea"), ("Eukaryota", "Archaea"), ("Eukaryota", "Bacteria")]
    overlap = {f"overlap_{a}_{b}": round(100 * _bhattacharyya(*stats[a], *stats[b]), 1)
               for a, b in pairs}
    for k, v in overlap.items():
        df[k] = v
    return df


def run(cfg, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else cfg.stage_output("pca")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_analysis_table(cfg.analysis_table, domains_only=True)
    coords, loadings, explained = fit(df)
    loadings.to_csv(out_dir / "pca_loadings.csv", index=False)
    coords.to_csv(out_dir / "pca_coordinates.csv", index=False)
    comp = compactness(coords)
    comp.to_csv(out_dir / "compactness_pairwise.csv", index=False)
    var = pd.DataFrame({"PC": ["PC1", "PC2"], "explained_variance_pct": (explained * 100).round(3)})
    var.to_csv(out_dir / "pca_variance.csv", index=False)
    print(f"[pca] PC1={explained[0]*100:.1f}% PC2={explained[1]*100:.1f}% "
          f"(cumulative {explained.sum()*100:.1f}%) -> {out_dir}")
    crosscheck_gam(cfg, comp, out_dir)
    return {"coords": coords, "loadings": loadings, "explained": explained, "compactness": comp}


def crosscheck_gam(cfg, comp: pd.DataFrame, out_dir: Path) -> float | None:
    """Emit the deposited GAM deviance (Table S15, R/mgcv) and cross-check that the
    Python PCA matches the R notebook via the per-domain compactness overlaps.
    Returns the max overlap difference, or None if the deposited outputs are absent."""
    import shutil
    dep = cfg.root / "00_data" / "pca_gam"
    gam = dep / "gam_deviance.csv"
    if gam.exists():
        shutil.copy2(gam, out_dir / "gam_deviance.csv")     # Table S15 (R-only)
    rcomp = dep / "compactness_R.csv"
    if not rcomp.exists():
        print("[pca] (no deposited R outputs to cross-check; run the R notebook for Table S15)")
        return None
    rc = pd.read_csv(rcomp).set_index("pair")["bhattacharyya_overlap"]
    got = {"Eukaryota-vs-Bacteria": comp["overlap_Eukaryota_Bacteria"].iloc[0] / 100,
           "Eukaryota-vs-Archaea": comp["overlap_Eukaryota_Archaea"].iloc[0] / 100,
           "Bacteria-vs-Archaea": comp["overlap_Bacteria_Archaea"].iloc[0] / 100}
    maxdiff = max(abs(float(rc[k]) - got[k]) for k in got)
    print(f"[pca] cross-check vs R notebook (Table S14/compactness): "
          f"max overlap diff = {maxdiff:.2e} ({'MATCH' if maxdiff < 1e-3 else 'MISMATCH'})")
    return maxdiff
