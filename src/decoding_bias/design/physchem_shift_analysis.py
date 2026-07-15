"""
physchem_shift_analysis.py - per-property WT->design shift analysis.

COMPLEMENTARY to the PCA projection:
  - PCA  = multivariate/geometric summary (where designs move, how far).
  - this = univariate decomposition (WHICH named properties drive the move,
           with effect sizes + significance).

Design (better than the preprint, which had no formal stats):
  - Aggregate the 8 design replicates per protein -> per-protein mean.
  - Pair against that protein's (same-predictor) WT value -> Delta.
  - Across the 25 proteins: paired Cohen's d_z = mean(Delta)/sd(Delta),
    Wilcoxon signed-rank p, BH-FDR across all model x property tests.
  - Model x property heatmap of d_z; optional per-domain breakdown.

Inputs: designs_features.csv + wt_features.csv  (from extract_design_features.py)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "design"))          # features_for_designs lives here
from features_for_designs import MIXED_FEATURES

PROPERTIES = MIXED_FEATURES  # decompose the SAME features the PCA uses


def per_protein_delta(designs: pd.DataFrame, wt: pd.DataFrame,
                      properties=PROPERTIES) -> pd.DataFrame:
    """For each (model, protein, property): mean(design) - WT, one row per
    (model, uniprot_id) with a column per property."""
    wt_idx = wt.set_index("uniprot_id")
    # per-protein mean over the 8 replicates
    dmean = designs.groupby(["model", "uniprot_id", "domain"])[properties].mean().reset_index()
    rows = []
    for r in dmean.itertuples(index=False):
        d = dict(model=r.model, uniprot_id=r.uniprot_id, domain=r.domain)
        if r.uniprot_id not in wt_idx.index:
            continue
        for p in properties:
            d[p] = getattr(r, p) - wt_idx.loc[r.uniprot_id, p]
        rows.append(d)
    return pd.DataFrame(rows)


def cohens_dz(delta: np.ndarray):
    """Paired effect size + Wilcoxon p for a vector of per-protein deltas."""
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    if n < 3 or np.allclose(delta, 0):
        return np.nan, np.nan, n, np.nan
    sd = delta.std(ddof=1)
    dz = delta.mean() / sd if sd > 0 else np.nan
    try:
        _, p = stats.wilcoxon(delta)
    except ValueError:
        p = np.nan
    return dz, p, n, delta.mean()


def effect_table(delta_df: pd.DataFrame, properties=PROPERTIES,
                 by_domain=False) -> pd.DataFrame:
    """Cohen's d_z + Wilcoxon p + BH-FDR per (model[, domain], property)."""
    group_cols = ["model", "domain"] if by_domain else ["model"]
    out = []
    for keys, g in delta_df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for p in properties:
            dz, pval, n, mean_d = cohens_dz(g[p].values)
            rec = dict(zip(group_cols, keys))
            rec.update(property=p, cohens_dz=dz, mean_delta=mean_d,
                       wilcoxon_p=pval, n_proteins=n)
            out.append(rec)
    res = pd.DataFrame(out)
    # BH-FDR across all tests in this table
    mask = res["wilcoxon_p"].notna()
    res["p_fdr"] = np.nan
    if mask.sum() > 0:
        from scipy.stats import false_discovery_control
        try:
            res.loc[mask, "p_fdr"] = false_discovery_control(res.loc[mask, "wilcoxon_p"].values)
        except Exception:
            # manual BH fallback
            pv = res.loc[mask, "wilcoxon_p"].values
            order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(pv)+1)
            res.loc[mask, "p_fdr"] = np.minimum.accumulate(
                (pv * len(pv) / ranks)[np.argsort(order)][::-1])[::-1]
    return res


def plot_heatmap(eff: pd.DataFrame, out_png, title="WT→design shift (Cohen's d_z)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat = eff.pivot(index="property", columns="model", values="cohens_dz").reindex(PROPERTIES)
    sig = eff.pivot(index="property", columns="model", values="p_fdr").reindex(PROPERTIES)
    fig, ax = plt.subplots(figsize=(1.4*mat.shape[1]+3, 0.45*mat.shape[0]+2))
    vmax = np.nanmax(np.abs(mat.values)) or 1
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=40, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v): continue
            s = sig.values[i, j]
            star = "***" if s < .001 else "**" if s < .01 else "*" if s < .05 else ""
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax*0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Cohen's d_z  (design − WT)")
    ax.set_title(title, fontsize=11); fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote", out_png)


def run(designs_csv=None, wt_csv=None, out_dir=None):
    designs_csv = designs_csv or REPO / "design" / "outputs" / "designs_features.csv"
    wt_csv      = wt_csv      or REPO / "design" / "outputs" / "wt_features.csv"
    out_dir     = out_dir     or REPO / "design" / "outputs"
    out = Path(out_dir); out.mkdir(exist_ok=True)
    designs = pd.read_csv(designs_csv); wt = pd.read_csv(wt_csv)
    delta = per_protein_delta(designs, wt)
    delta.to_csv(out / "wt_design_deltas.csv", index=False)

    eff = effect_table(delta, by_domain=False)
    eff.to_csv(out / "physchem_effect_sizes.csv", index=False)
    eff_dom = effect_table(delta, by_domain=True)
    eff_dom.to_csv(out / "physchem_effect_sizes_by_domain.csv", index=False)

    plot_heatmap(eff, out / "physchem_shift_heatmap.png")
    print(f"\nTop shifts (|d_z|, FDR<0.05):")
    sig = eff[(eff.p_fdr < 0.05)].reindex(eff.cohens_dz.abs().sort_values(ascending=False).index)
    print(sig[["model", "property", "cohens_dz", "p_fdr"]].head(12).to_string(index=False))
    return eff, eff_dom, delta


if __name__ == "__main__":
    run()
