"""
physchem_importance_table_figure.py - assemble the main-dataset property->score
importance analysis into ONE results table + ONE figure.

Inputs (already computed by property_importance.py / property_importance_baseline.py):
  property_importance.csv               univariate beta, multivariate beta, mv_p, VIF,
                                         Johnson's relative weight %, model_R2, group
  property_importance_baseline_adjusted.csv  composition-baseline-adjusted rel weights
  property_collinearity_clusters.csv    correlation-cluster id per property

Outputs (design/outputs/):
  physchem_importance_master.csv        merged per (model x property) results table
  physchem_relweight_wide.csv           Johnson's RW% pivot (property x model)
  fig_physchem_importance.png           3-panel: univariate b | multivariate b (+VIF/sig)
                                         | Johnson's RW%   (the three-layer story)

The three layers, read together: univariate = marginal association; multivariate =
joint coefficients (shrink / sign-flip under collinearity, so VIF>5 cells are flagged
and not trusted alone); Johnson's relative weights = collinearity-robust share of model
R^2 -> the importance we report. Sequence-only models are circularity-flagged
(composition<->likelihood is partly mechanical) and shown in a separate block.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = HERE / "outputs"

SEQ = ["sequence_length","mw_per_residue","isoelectric_point",
       "acidic_residue_fraction","basic_residue_fraction","gravy","aromaticity",
       "instability_index","proline_fraction"]                      # 9 (14-feature set)
STRUCT = ["ordered_percent","helix_sheet_contrast","rco","avg_cb_distance","surface_exposure"]
PROP_ORDER = SEQ + STRUCT                                           # 14
STRUCT_MODELS = ["ProteinMPNN","SolubleMPNN","Caliby","SolubleCaliby","ESM-IF",
                 "MIF","MIF-ST","TriFlow","ESM3-struct"]
SEQ_MODELS = ["ESM2-15B","ESM3-seq","CARP-640M","ProGen2","ProGen2-XL","ProtGPT2"]
MODEL_ORDER = STRUCT_MODELS + SEQ_MODELS                            # 15
PRETTY = {p: p.replace("_"," ") for p in PROP_ORDER}


def build_master():
    # Prefer the expanded 15-model / 14-feature run; fall back to the legacy file.
    src = OUT/"property_importance_expanded.csv"
    m = pd.read_csv(src if src.exists() else OUT/"property_importance.csv")
    # Optional enrichments (only present for the legacy run); merge if available.
    adj_f = OUT/"property_importance_baseline_adjusted.csv"
    if adj_f.exists():
        adj = pd.read_csv(adj_f)[["model","property","rel_weight_adj","baseline_R2","R2_adj"]]
        m = m.merge(adj, on=["model","property"], how="left")
    clu_f = OUT/"property_collinearity_clusters.csv"
    if clu_f.exists():
        m = m.merge(pd.read_csv(clu_f), on="property", how="left")
    m["feature_class"] = np.where(m.property.isin(SEQ), "sequence", "structure")
    m = m[m.model.isin(MODEL_ORDER)]                      # drop fine-tuned / extra rows
    m["model"] = pd.Categorical(m.model, [x for x in MODEL_ORDER if x in m.model.unique()], ordered=True)
    m["property"] = pd.Categorical(m.property, PROP_ORDER, ordered=True)
    return m.sort_values(["model","property"])


def grid(m, value):
    return m.pivot(index="property", columns="model", values=value).reindex(
        index=PROP_ORDER, columns=[c for c in MODEL_ORDER if c in m.model.unique()])


def main():
    m = build_master()
    m.to_csv(OUT/"physchem_importance_master.csv", index=False)
    rw = grid(m, "rel_weight_pct"); rw.to_csv(OUT/"physchem_relweight_wide.csv")
    uni = grid(m, "univariate_beta"); mv = grid(m, "multivariate_beta")
    vif = grid(m, "VIF"); pval = grid(m, "mv_p")

    models = list(rw.columns); nC = len(models); nR = len(PROP_ORDER)
    sep = len([x for x in STRUCT_MODELS if x in models]) - 0.5   # struct|seq divider

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    bmax = np.nanpercentile(np.abs(np.r_[uni.values.ravel(), mv.values.ravel()]), 98)

    def heat(ax, data, cmap, vmin, vmax, title, cbar_label):
        im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(nC)); ax.set_xticklabels(models, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(nR)); ax.set_yticklabels([PRETTY[p] for p in PROP_ORDER], fontsize=8)
        ax.axvline(sep, color="k", lw=1.4)
        ax.axhline(len(SEQ)-0.5, color="grey", lw=0.8, ls=":")   # seq|struct feature divider
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.6, label=cbar_label)
        return im

    # Panel A: univariate beta
    heat(axes[0], uni, "RdBu_r", -bmax, bmax,
         "(A) Univariate  (marginal assoc.)", "standardised beta")
    # Panel B: multivariate beta, with VIF>5 boxed + significant cells dotted
    heat(axes[1], mv, "RdBu_r", -bmax, bmax,
         "(B) Multivariate  (joint; collinearity-affected)", "standardised beta")
    for i, p in enumerate(PROP_ORDER):
        for j, mod in enumerate(models):
            if vif.loc[p, mod] is not None and vif.loc[p, mod] > 5:   # high collinearity -> distrust
                axes[1].add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                            edgecolor="black", lw=1.3, hatch="///"))
            if pd.notna(pval.loc[p, mod]) and pval.loc[p, mod] < 0.05:
                axes[1].plot(j, i, ".", color="k", ms=2)
    # Panel C: Johnson's relative weights
    heat(axes[2], rw, "magma", 0, np.nanpercentile(rw.values, 98),
         "(C) Johnson's relative weight  (robust importance)", "% of model R^2")

    fig.suptitle("Physicochemical property -> model-score importance (main dataset; "
                 "left of bar = structure-conditioned, right = sequence-only/circularity-flagged)\n"
                 "Panel B: hatched = VIF>5 (collinear, distrust single beta); dot = p<0.05",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT/"fig_physchem_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- console summary: top-3 properties per model by Johnson's weight ----
    print("Top-3 properties by Johnson's relative weight (% of model R^2):")
    for mod in models:
        s = m[m.model == mod].nlargest(3, "rel_weight_pct")
        bits = ", ".join(f"{r.property} {r.rel_weight_pct:.0f}%" for r in s.itertuples())
        print(f"  {mod:14s} R2={m[m.model==mod].model_R2.iloc[0]:.2f} | {bits}")
    print(f"\nWrote physchem_importance_master.csv, physchem_relweight_wide.csv, "
          f"fig_physchem_importance.png")


if __name__ == "__main__":
    main()
