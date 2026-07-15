"""
Publication-quality vector PCA figures (PDF + PNG), one focused view per file,
in the style of the manuscript Fig.4 (grey natural cloud, per-model 2σ ellipses,
WT marked ✕, WT→design shift arrows; designs coloured by model).

Outputs to design/outputs/pca_figures/:
  overview_by_model.pdf            all designs over the cloud, coloured by model
  overview_by_domain.pdf           all designs over the cloud, coloured by domain
  centroid_shifts.pdf              global WT-mean → per-model centroid vectors
  per_model/<model>.pdf            one model: designs + per-protein WT→mean arrows
  per_protein/<uid>.pdf            one protein: designs by model + 2σ ellipses + WT
  per_protein_grid.pdf             all 25 proteins as small multiples
Built from the pca_explorer_build bundle (16-feature PCA, calibrated designs).
"""
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
from scipy.stats import f as fdist, chi2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = HERE / "outputs"; FIG = OUT / "pca_figures"
(FIG/"per_model").mkdir(parents=True, exist_ok=True)
(FIG/"per_protein").mkdir(parents=True, exist_ok=True)
(FIG/"functions").mkdir(parents=True, exist_ok=True)
(FIG/"centroid_shifts").mkdir(parents=True, exist_ok=True)

DOM = {"Archaea": "#E41A1C", "Bacteria": "#1976D2", "Eukaryota": "#388E3C"}
MODELS = ["ProteinMPNN","SolubleMPNN","Caliby","SolubleCaliby","ESM-IF","MIF","MIF-ST"]
MCOL = {"ProteinMPNN":"#1F77B4","SolubleMPNN":"#7FB7E0","Caliby":"#D62728",
        "SolubleCaliby":"#FF9896","ESM-IF":"#FF7F0E","MIF":"#9467BD","MIF-ST":"#2CA02C"}
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8,
                     "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})


def varexp():
    t = (OUT/"pca_axis_interpretation.txt").read_text()
    m = re.search(r"PC1 (\d+)%, PC2 (\d+)%", t)
    return (int(m.group(1)), int(m.group(2))) if m else (24, 17)
VX = varexp()
XL, YL = f"PC1 ({VX[0]}%)", f"PC2 ({VX[1]}%)"


def conf_ellipse(ax, pts, color, level=0.95, kind="data", lw=1.5, ls="-", fill=False):
    """Small-sample-correct 2D ellipse.
      kind='data' : region expected to contain `level` of the points (bivariate-normal
                    PREDICTION ellipse, F-distribution scaling -> correct at small n;
                    -> chi-square as n grows).
      kind='mean' : `level` CONFIDENCE region for the centroid (Hotelling T^2 / n).
    Falls back to chi-square for large n. Skips if n<3 (cov undefined)."""
    pts = np.asarray(pts, float); pts = pts[np.isfinite(pts).all(1)]
    n = len(pts)
    if n < 3: return
    mu = pts.mean(0); cov = np.cov(pts.T)
    val, vec = np.linalg.eigh(cov); order = val.argsort()[::-1]
    val, vec = val[order], vec[:, order]
    theta = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
    if n > 60:                                   # large sample -> chi-square
        r2 = chi2.ppf(level, 2) / (n if kind == "mean" else 1)
    else:                                        # small sample -> F-distribution
        base = 2 * (n - 1) / (n - 2) * fdist.ppf(level, 2, n - 2)
        r2 = base / (n if kind == "mean" else 1)
    w, h = 2 * np.sqrt(np.clip(val, 0, None) * r2)
    ax.add_patch(Ellipse(mu, w, h, angle=theta, fill=fill, facecolor=color if fill else "none",
                         alpha=0.12 if fill else 1, edgecolor=color, lw=lw, ls=ls))


# back-compat alias used below
def cov_ellipse(ax, pts, color, nstd=2.0, lw=1.5, ls="-"):
    conf_ellipse(ax, pts, color, level=0.95, kind="data", lw=lw, ls=ls)


def base_ax(ax, cloud, xr=None, yr=None, cloud_s=2, cloud_a=0.25):
    ax.scatter(cloud.PC1, cloud.PC2, s=cloud_s, c="lightgrey", alpha=cloud_a, linewidths=0, zorder=0)
    ax.axhline(0, color="grey", lw=0.6, ls="--", zorder=1)
    ax.axvline(0, color="grey", lw=0.6, ls="--", zorder=1)
    ax.set_xlabel(XL); ax.set_ylabel(YL)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if xr: ax.set_xlim(xr)
    if yr: ax.set_ylim(yr)


def model_legend(fig, models, ncol=None, y=-0.04):
    h = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=MCOL[m],
                    markersize=8, label=m) for m in models]
    fig.legend(handles=h, loc="lower center", ncol=ncol or len(models),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, y))


def main():
    cloud = pd.read_csv(OUT/"pca_cloud.csv")
    des = pd.read_csv(OUT/"pca_designs.csv")
    wt = pd.read_csv(OUT/"pca_wt.csv").set_index("uniprot_id")
    cent = des.groupby(["model","uniprot_id"]).agg(PC1=("PC1","mean"),PC2=("PC2","mean")).reset_index()
    XR = (cloud.PC1.min()-0.5, cloud.PC1.max()+0.5)
    YR = (cloud.PC2.min()-0.5, cloud.PC2.max()+0.5)

    def save(fig, name):
        fig.savefig(FIG/f"{name}.pdf"); fig.savefig(FIG/f"{name}.png", dpi=200); plt.close(fig)

    # 1. overview by model
    fig, ax = plt.subplots(figsize=(7,6.5))
    base_ax(ax, cloud, XR, YR)
    for m in MODELS:
        s = des[des.model==m]
        ax.scatter(s.PC1, s.PC2, s=8, c=MCOL[m], alpha=0.55, linewidths=0, zorder=2)
    ax.scatter(wt.PC1, wt.PC2, marker="x", c="black", s=45, lw=1.2, zorder=4)
    ax.set_title("All designs in WT PCA space, by model")
    model_legend(fig, MODELS); save(fig, "overview_by_model")

    # 1b. overview by model WITH per-model 95% data ellipses
    fig, ax = plt.subplots(figsize=(7,6.5))
    base_ax(ax, cloud, XR, YR)
    for m in MODELS:
        s = des[des.model==m]
        ax.scatter(s.PC1, s.PC2, s=8, c=MCOL[m], alpha=0.45, linewidths=0, zorder=2)
        conf_ellipse(ax, s[["PC1","PC2"]].values, MCOL[m], level=0.95, kind="data", lw=1.8)
    ax.scatter(wt.PC1, wt.PC2, marker="x", c="black", s=45, lw=1.2, zorder=4)
    ax.set_title("All designs in WT PCA space, by model (95% data ellipse)")
    model_legend(fig, MODELS); save(fig, "overview_by_model_ellipses")

    # 2. overview by domain
    fig, ax = plt.subplots(figsize=(7,6.5))
    base_ax(ax, cloud, XR, YR)
    for d,c in DOM.items():
        s = des[des.domain==d]; ax.scatter(s.PC1, s.PC2, s=8, c=c, alpha=0.5, linewidths=0, zorder=2)
        cov_ellipse(ax, s[["PC1","PC2"]].values, c)
    ax.scatter(wt.PC1, wt.PC2, marker="x", c="black", s=45, lw=1.2, zorder=4)
    ax.set_title("All designs in WT PCA space, by domain (2σ)")
    h=[plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=c,markersize=8,label=d) for d,c in DOM.items()]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5,-0.02))
    save(fig, "overview_by_domain")

    # 3. centroid shifts - scaled to the centroid region (with a little cloud context)
    def centroid_fig(sub_des, sub_wt, title, fname):
        wtmean = sub_wt[["PC1","PC2"]].mean().values
        cen = sub_des.groupby("model").agg(PC1=("PC1","mean"),PC2=("PC2","mean"))
        pts = np.vstack([cen.values, wtmean])
        pad = max(0.6, 0.35*np.ptp(pts, axis=0).max())
        xr = (pts[:,0].min()-pad, pts[:,0].max()+pad); yr = (pts[:,1].min()-pad, pts[:,1].max()+pad)
        fig, ax = plt.subplots(figsize=(6.5,6))
        base_ax(ax, cloud, xr, yr, cloud_s=3, cloud_a=0.18)
        for m in MODELS:
            if m not in cen.index: continue
            c = cen.loc[m].values
            ax.annotate("", xy=c, xytext=wtmean,
                arrowprops=dict(arrowstyle="-|>", color=MCOL[m], lw=2.4, shrinkA=0, shrinkB=0), zorder=3)
            ax.scatter(*c, s=70, c=MCOL[m], zorder=4, edgecolors="white", linewidths=0.9)
        ax.scatter(*wtmean, marker="x", c="black", s=90, lw=2.2, zorder=5)
        ax.set_title(title)
        model_legend(fig, MODELS); save(fig, fname)
    cent_des = des.copy()
    centroid_fig(cent_des, wt.reset_index(), "Model centroid shifts in WT PCA space",
                 "centroid_shifts/all")
    for dom in ["Archaea","Bacteria","Eukaryota"]:
        sd = des[des.domain==dom]; sw = wt[wt.domain==dom].reset_index()
        if len(sw) >= 2:
            centroid_fig(sd, sw, f"Model centroid shifts - {dom}", f"centroid_shifts/{dom}")

    # 4. per model: designs + per-protein WT->mean arrows + ellipse
    for m in MODELS:
        s = des[des.model==m]; mn = cent[cent.model==m]
        fig, ax = plt.subplots(figsize=(7,6.5))
        base_ax(ax, cloud, XR, YR)
        ax.scatter(s.PC1, s.PC2, s=8, c=MCOL[m], alpha=0.4, linewidths=0, zorder=2)
        for _,r in mn.iterrows():
            if r.uniprot_id in wt.index:
                w = wt.loc[r.uniprot_id]
                ax.annotate("", xy=(r.PC1,r.PC2), xytext=(w.PC1,w.PC2),
                    arrowprops=dict(arrowstyle="-|>", color=MCOL[m], lw=1, alpha=0.7, shrinkA=0, shrinkB=0), zorder=3)
        ax.scatter(mn.PC1, mn.PC2, s=40, c=MCOL[m], edgecolors="white", linewidths=0.8, zorder=4)
        cov_ellipse(ax, s[["PC1","PC2"]].values, MCOL[m])
        ax.scatter(wt.PC1, wt.PC2, marker="x", c="black", s=30, lw=1, alpha=0.5, zorder=4)
        ax.set_title(f"{m}: designs + WT→mean shifts")
        save(fig, f"per_model/{m}")

    # 5. per protein: designs by model + per-model 2σ + WT (FIXED full PCA space)
    for uid in sorted(des.uniprot_id.unique()):
        sp = des[des.uniprot_id==uid]; w = wt.loc[uid]
        dom, spec = sp.domain.iloc[0], (w.species if "species" in w else "")
        fig, ax = plt.subplots(figsize=(6.5,6))
        base_ax(ax, cloud, XR, YR, cloud_s=2, cloud_a=0.25)
        for m in MODELS:
            g = sp[sp.model==m]
            if not len(g): continue
            ax.scatter(g.PC1, g.PC2, s=14, c=MCOL[m], alpha=0.7, linewidths=0, zorder=2)
            cov_ellipse(ax, g[["PC1","PC2"]].values, MCOL[m], lw=1.2)
            ax.scatter(g.PC1.mean(), g.PC2.mean(), s=45, c=MCOL[m], edgecolors="white", linewidths=0.8, zorder=4)
        ax.scatter([w.PC1],[w.PC2], marker="x", c="black", s=70, lw=2, zorder=5)
        ax.set_title(f"{uid} - {str(spec)[:38]}, {dom}", fontsize=11)
        fig.subplots_adjust(bottom=0.18)
        model_legend(fig, MODELS, ncol=4, y=-0.12); save(fig, f"per_protein/{uid}")

    # 6. main-dataset broad-function clusters - one separate graph per function
    funcs = cloud.broad_function.value_counts()
    funcs = funcs[funcs >= 30].head(12).index
    for fn in funcs:
        s = cloud[cloud.broad_function == fn]
        fig, ax = plt.subplots(figsize=(7,6.5))
        base_ax(ax, cloud, XR, YR)
        for d, col in DOM.items():
            sd = s[s.domain == d]
            ax.scatter(sd.PC1, sd.PC2, s=10, c=col, alpha=0.6, linewidths=0, zorder=2)
        conf_ellipse(ax, s[["PC1","PC2"]].values, "black", level=0.95, kind="data", lw=2)
        ax.set_title(f"{fn}  (n={len(s)}; 95% data ellipse, colour = domain)")
        h=[plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=col,markersize=8,label=d) for d,col in DOM.items()]
        fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5,-0.02))
        save(fig, f"functions/{fn}")

    prot = sorted(des.uniprot_id.unique())
    n = 4 + 3 + len(MODELS) + len(prot) + len(funcs)
    print(f"Wrote {n} vector figures (PDF+PNG) to {FIG}/")
    print("  overview_by_model(+_ellipses), overview_by_domain, centroid_shifts/ (all+3 domains),")
    print(f"  per_model/ ({len(MODELS)}), per_protein/ ({len(prot)}), functions/ ({len(funcs)})")


if __name__ == "__main__":
    main()
