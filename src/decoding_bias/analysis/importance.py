"""Property-to-score importance (SI Fig S6; Tables S16-S18).

Per model: within-family de-mean (controls protein type), standardise, then
Johnson relative weights (% of within-family R^2, collinearity-robust), univariate
within-family beta, multivariate beta with species-clustered SE, and VIF.
Deterministic (no RNG). The 15-model panel and the fine-tuned-020 arm are the
`catalog` registries; the feature set is the pruned 14.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .. import catalog
from ..data import load_analysis_table

SEQ = catalog.SEQUENCE_FEATURES_14
STRUCT = catalog.STRUCTURE_FEATURES_14
FEATS = catalog.BIOPHYS_14
PANEL = catalog.IMPORTANCE_PANEL
FINETUNE = catalog.FINETUNE_020


def within_family_demean(df, cols, fam="protein_family"):
    return df[cols] - df.groupby(fam)[cols].transform("mean")


def johnson_relative_weights(X, y):
    Rxx = np.corrcoef(X, rowvar=False)
    rxy = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    w, V = np.linalg.eigh(Rxx)
    w = np.clip(w, 1e-8, None)
    Lam = V @ np.diag(np.sqrt(w)) @ V.T
    beta = np.linalg.solve(Lam, rxy)
    eps = (Lam ** 2) @ (beta ** 2)
    R2 = eps.sum()
    return eps / R2 * 100, R2


def vif(X):
    return np.diag(np.linalg.pinv(np.corrcoef(X, rowvar=False)))


def analyse(df, models, label, min_rows=200):
    rows = []
    for model, col in models.items():
        if col not in df.columns:
            print(f"  [skip] {model}: column {col} absent")
            continue
        d = df.dropna(subset=[col, "protein_family", "species"] + FEATS).copy()
        if len(d) < min_rows:
            print(f"  [skip] {model}: only {len(d)} complete rows")
            continue
        dm = within_family_demean(d, FEATS + [col])
        dm = (dm - dm.mean()) / dm.std()
        X = dm[FEATS].values
        y = dm[col].values
        relw, R2 = johnson_relative_weights(X, y)
        vifs = vif(X)
        dd = d.copy()
        for f in FEATS:
            dd[f] = dm[f].values
        dd["_y"] = y
        m = smf.ols("_y ~ " + " + ".join(FEATS), data=dd).fit(
            cov_type="cluster", cov_kwds={"groups": dd["species"]})
        for j, f in enumerate(FEATS):
            uni = np.corrcoef(X[:, j], y)[0, 1]
            rows.append(dict(model=model, group=label, property=f, n=len(d),
                             univariate_beta=uni, multivariate_beta=m.params.get(f, np.nan),
                             mv_p=m.pvalues.get(f, np.nan), VIF=vifs[j],
                             rel_weight_pct=relw[j], model_R2=R2,
                             feature_class=("sequence" if f in SEQ else "structure")))
    return pd.DataFrame(rows)


def run(cfg, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir) if out_dir else cfg.stage_output("property_importance")
    out_dir.mkdir(parents=True, exist_ok=True)
    min_rows = cfg.params("property_importance").get("min_rows", 200)

    v = load_analysis_table(cfg.analysis_table, domains_only=True)
    print(f"[importance] input n={len(v)}")
    panel = analyse(v, PANEL, "panel", min_rows)
    ft = analyse(v, FINETUNE, "finetune_020", min_rows)
    allres = pd.concat([panel, ft], ignore_index=True)
    allres.to_csv(out_dir / "property_importance_expanded.csv", index=False)

    wide = (panel.pivot(index="property", columns="model", values="rel_weight_pct")
                 .reindex(FEATS)[list(PANEL)])
    wide.to_csv(out_dir / "physchem_relweight_wide_expanded.csv")
    wide_ft = (ft.pivot(index="property", columns="model", values="rel_weight_pct")
                 .reindex(FEATS)[list(FINETUNE)])
    wide_ft.to_csv(out_dir / "physchem_relweight_wide_finetune020.csv")
    print(f"[importance] {allres['model'].nunique()} models -> {out_dir}")
    return allres
