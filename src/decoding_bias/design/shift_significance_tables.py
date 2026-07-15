"""
shift_significance_tables.py - per-(protein, model, property) WT->design shift
significance tables.

Finer-grained than physchem_shift_analysis.py (which tests shifts ACROSS the 25
proteins). Here, for every (protein x model x property) cell we test that
protein-model's 8 design replicates against its wild-type value:

  starting value = WT property value (ColabFold, same predictor as the designs)
  design value   = mean of the 8 design replicates
  mean shift     = design value - starting value          (native units)
  shift_z        = mean shift / SD(property across the 25 WTs)  (comparable units)
  margin         = 95% CI half-width on the mean shift = t(.975,7)*sd_rep/sqrt(8)
  d_z            = mean shift / SD(the 8 replicate values)  (within-cell effect)
  p              = one-sample t-test of (replicate - WT) vs 0
  p_wilcox       = Wilcoxon signed-rank (robustness; n=8)
  p_fdr          = Benjamini-Hochberg across all cells
  sig            = p_fdr < 0.05

Outputs (design/outputs/):
  shift_significance_master.csv         long table, all 25x7x16 cells
  shift_table_protein_<Entry>.csv       protein-focused view (property x model)
  shift_table_model_<Model>.csv         model-focused view (protein x property)
  shift_significance_summary.csv        # significant cells per model x property
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

OUT = REPO / "design" / "outputs"                 # read designs/wt + write tables here
# Two of the 16 PCA features carry no per-design signal and are dropped from the
# shift tables: sequence_length is fixed (full redesign keeps the backbone length,
# shift == 0 always) and surface_exposure is near-constant by its percentile
# definition (~0.30 for every chain, between-protein SD ~0.002). Keeping them
# would only inject trivial/degenerate rows.
DEGENERATE = ["sequence_length", "surface_exposure"]
PROPERTIES = [p for p in MIXED_FEATURES if p not in DEGENERATE]   # 14 informative features
T_CRIT_7 = stats.t.ppf(0.975, df=7)   # 95% CI multiplier for n=8


def build_master(designs: pd.DataFrame, wt: pd.DataFrame) -> pd.DataFrame:
    wt_idx = wt.set_index("uniprot_id")
    # between-protein SD per property (WTs; same predictor as designs) -> z-scale
    pop_sd = {p: wt[p].std(ddof=1) for p in PROPERTIES}
    rows = []
    for (uid, model), g in designs.groupby(["uniprot_id", "model"]):
        if uid not in wt_idx.index:
            continue
        domain = g["domain"].iloc[0]
        for p in PROPERTIES:
            vals = g[p].dropna().values
            wt_val = wt_idx.loc[uid, p]
            if len(vals) < 3 or not np.isfinite(wt_val):
                continue
            design_mean = vals.mean()
            shift = design_mean - wt_val
            sd_rep = vals.std(ddof=1)
            shifts = vals - wt_val
            # one-sample t-test of replicate shifts vs 0
            if sd_rep > 0:
                t, p_t = stats.ttest_1samp(shifts, 0.0)
                dz = shift / sd_rep
                margin = T_CRIT_7 * sd_rep / np.sqrt(len(vals))
            else:
                # zero replicate variance: a constant cell. No shift -> not
                # significant (p=1); a constant non-zero shift is degenerate (skip p).
                t, dz, margin = np.nan, (0.0 if shift == 0 else np.inf), 0.0
                p_t = 1.0 if np.isclose(shift, 0.0) else np.nan
            # Wilcoxon (robustness); undefined if all shifts identical/zero
            try:
                _, p_w = stats.wilcoxon(shifts)
            except ValueError:
                p_w = np.nan
            rows.append(dict(
                Entry=uid, model=model, domain=domain, property=p,
                starting_value=round(float(wt_val), 4),
                design_value=round(float(design_mean), 4),
                mean_shift=round(float(shift), 4),
                shift_z=round(float(shift / pop_sd[p]), 4) if pop_sd[p] else np.nan,
                margin=round(float(margin), 4),
                d_z=round(float(dz), 3),
                p=p_t, p_wilcox=p_w, n=len(vals)))
    m = pd.DataFrame(rows)
    # BH-FDR across all cells with a defined p
    ok = m["p"].notna()
    m.loc[ok, "p_fdr"] = _bh(m.loc[ok, "p"].values)
    m["sig"] = m["p_fdr"] < 0.05
    return m


def _bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def _fmt(x, sig=False):
    """+0.21* style: signed value with a star when significant."""
    if pd.isna(x):
        return ""
    return f"{x:+.3f}" + ("*" if sig else "")


def main():
    designs = pd.read_csv(OUT / "designs_features.csv")
    wt = pd.read_csv(OUT / "wt_features.csv")
    m = build_master(designs, wt)
    m.to_csv(OUT / "shift_significance_master.csv", index=False)

    # ---- protein-focused example: property (rows) x model (cols), mean shift +/- margin ----
    example_protein = designs["uniprot_id"].iloc[0]
    pf = m[m.Entry == example_protein].copy()
    pf["cell"] = pf.apply(lambda r: f"{r.mean_shift:+.3f} +/-{r.margin:.3f}{'*' if r.sig else ''}", axis=1)
    pf_tbl = pf.pivot(index="property", columns="model", values="cell").reindex(PROPERTIES)
    pf_tbl.to_csv(OUT / f"shift_table_protein_{example_protein}.csv")

    # ---- model-focused example: protein (rows) x property (cols), signed d_z with star ----
    example_model = "ProteinMPNN"
    mf = m[m.model == example_model].copy()
    mf["cell"] = mf.apply(lambda r: _fmt(r.shift_z, r.sig), axis=1)
    mf_tbl = mf.pivot(index="Entry", columns="property", values="cell").reindex(columns=PROPERTIES)
    mf_tbl.to_csv(OUT / f"shift_table_model_{example_model}.csv")

    # ---- summary: # significant cells per model x property ----
    summ = (m.groupby(["model", "property"])["sig"].agg(["sum", "count"])
              .reset_index().rename(columns={"sum": "n_sig", "count": "n_total"}))
    summ["frac_sig"] = (summ.n_sig / summ.n_total).round(3)
    summ_wide = summ.pivot(index="property", columns="model", values="n_sig").reindex(PROPERTIES)
    summ_wide.to_csv(OUT / "shift_significance_summary.csv")

    print(f"master: {len(m)} cells ({m.Entry.nunique()} proteins x "
          f"{m.model.nunique()} models x {m.property.nunique()} properties)")
    print(f"significant (FDR<0.05): {int(m.sig.sum())} / {len(m)} "
          f"({100*m.sig.mean():.0f}%)")
    print("\n# significant cells per model (out of 25 proteins x 16 properties = 400):")
    print(m.groupby("model")["sig"].sum().sort_values(ascending=False).to_string())
    print("\n# significant cells per property (out of 25 x 7 = 175):")
    print(m.groupby("property")["sig"].sum().sort_values(ascending=False).to_string())
    print(f"\nWrote shift_significance_master.csv, "
          f"shift_table_protein_{example_protein}.csv, "
          f"shift_table_model_{example_model}.csv, shift_significance_summary.csv")


if __name__ == "__main__":
    main()
