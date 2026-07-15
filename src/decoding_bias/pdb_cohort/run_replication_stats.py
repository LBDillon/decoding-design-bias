"""
run_replication_stats.py - does the per-model residual taxonomic bias REPLICATE across
diverse inputs (experimental-PDB cohort vs predicted-AFDB), with statistics?

Three inputs, all decomposed with the SAME per-model-complete-case method (14-feature
biophysics + protein family + species), restricted to the cohort's model panel:
    PDB     : 876 experimental chains, resolved-chain seqs   (cohort_score_variance_decomposition.csv)
    AF2m    : 876 AFDB, domain-matched, ribosomal-free, mean +/- sd over 30 draws (matched_af2_vd_control.csv)
    AF2full : ~10k AFDB (recomputed here on the cohort panel, ProGen2-XL, per-model complete case)

Tests:
  1. REPLICATION of the pattern  : Spearman (rank) + Pearson correlation of the per-model
     residual-species vector between inputs, with p-values (incl. a TriFlow-excluded variant,
     TriFlow being a flagged cohort outlier).
  2. PER-MODEL DEPARTURE          : z = (PDB - AF2m_mean)/AF2m_sd, two-sided p, BH-adjusted
     -> which models' PDB residual significantly exceeds the matched-AFDB baseline.
  3. PAIRED structure vs sequence : Wilcoxon signed-rank of (PDB - AF2m) within each class
     -> sequence models replicate (~0); structure models elevated on experimental structures.

Outputs (pdb_robustness/): data/vd_replication_stats.csv, data/vd_replication_correlations.csv,
tables/table_vd_replication.tex, figures/fig_vd_replication.{png,pdf}
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "design"))
import run_cohort_vd as RC          # reuse decompose_one + BIOPHYS + PANEL

IC = REPO / "design" / "outputs" / "independent_cohort"
MAIN = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12_cli.csv"
RB = REPO / "pdb_robustness"
(RB / "data").mkdir(parents=True, exist_ok=True); (RB / "tables").mkdir(exist_ok=True); (RB / "figures").mkdir(exist_ok=True)

PANEL = RC.MODELS                    # canonical -> (col, type)
TYPE_C = {"structure": "#2E7D32", "hybrid": "#F9A825", "hybrid-ST": "#C0CA33",
          "sequence": "#00838F", "structure(FT)": "#7E57C2"}


def af2_full_residuals(models):
    """Per-model residual species on the full AFDB dataset (same method as the cohort)."""
    df = pd.read_csv(MAIN, low_memory=False)
    df = df[df.domain.isin(["Archaea", "Bacteria", "Eukaryota"])].copy()
    cc = df[df[RC.BIOPHYS].notna().all(axis=1)].copy()
    Z = ((cc[RC.BIOPHYS] - cc[RC.BIOPHYS].mean()) / cc[RC.BIOPHYS].std()).values
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    cc["PC1"], cc["PC2"] = (U[:, 0] * S[0]), (U[:, 1] * S[1])
    out = {}
    for m in models:
        col, kind = PANEL[m]
        if col not in cc.columns or cc[col].notna().sum() < RC.MIN_N:
            continue
        rec = RC.decompose_one(cc, col, kind)
        if rec:
            out[m] = rec
    return out


def corr_block(a, b, names, label):
    """Pearson + Spearman with p, on aligned vectors a,b across the model panel."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return [dict(comparison=label, n_models=len(a),
                 pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp)]


def repel_labels(ax, xs, ys, labels, fontsize=7, n_iter=150):
    """Dependency-free text repel: put each label by its point, then iteratively push
    overlapping label boxes apart (in display coords) and draw a thin leader line back
    to the marker. Avoids the overlapping model names in the replication scatter."""
    fig = ax.figure
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    texts = [ax.text(x, y, s, fontsize=fontsize, zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
             for x, y, s in zip(xs, ys, labels)]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))   # marker positions (display px)
    pos = pts + np.array([7.0, 7.0])                           # start labels up-right of points
    inv = ax.transData.inverted()
    for _ in range(n_iter):
        for t, p in zip(texts, pos):
            t.set_position(inv.transform(p))
        fig.canvas.draw()
        bb = [t.get_window_extent(rend) for t in texts]
        moved = False
        for i in range(len(texts)):
            push = np.zeros(2)
            for j in range(len(texts)):
                if i != j and bb[i].overlaps(bb[j]):
                    ci = np.array([(bb[i].x0 + bb[i].x1) / 2, (bb[i].y0 + bb[i].y1) / 2])
                    cj = np.array([(bb[j].x0 + bb[j].x1) / 2, (bb[j].y0 + bb[j].y1) / 2])
                    diff = ci - cj; norm = np.hypot(*diff) or 1.0
                    push += diff / norm * 3.0
            d0 = pos[i] - pts[i]; n0 = np.hypot(*d0) or 1.0    # keep the box off its own marker
            if n0 < 12:
                push += d0 / n0 * 2.0
            if push.any():
                pos[i] += push; moved = True
        if not moved:
            break
    for t, p0 in zip(texts, pts):
        x, y = t.get_position()
        x0, y0 = inv.transform(p0)
        ax.plot([x0, x], [y0, y], color="0.6", lw=0.4, zorder=5)


def main():
    # ---- gather per-model residuals from the three inputs --------------------
    cohort = pd.read_csv(IC / "cohort_score_variance_decomposition.csv").set_index("model")
    af2m = pd.read_csv(IC / "matched_af2_vd_control.csv").set_index("model")
    models = [m for m in PANEL if m in cohort.index and m in af2m.index]
    af2full = af2_full_residuals(models)

    rows = []
    for m in models:
        col, kind = PANEL[m]
        rec = dict(model=m, type=kind,
                   PDB_resid_raw=cohort.loc[m, "dSpecies_given_family_biophys"],
                   PDB_resid_adj=cohort.loc[m, "residual_species_R2_adj"],
                   PDB_retention=cohort.loc[m, "species_effect_retention_given_family_biophys"],
                   PDB_species_p=cohort.loc[m, "species_p_given_family_biophys"],
                   AF2m_resid_raw_mean=af2m.loc[m, "AF2_resid_raw_mean"],
                   AF2m_resid_raw_sd=af2m.loc[m, "AF2_resid_raw_sd"],
                   AF2m_resid_adj_mean=af2m.loc[m, "AF2_resid_adj_mean"],
                   AF2m_retention_mean=af2m.loc[m, "AF2_retention_mean"])
        if m in af2full:
            rec["AF2full_resid_raw"] = af2full[m]["dSpecies_given_family_biophys"]
            rec["AF2full_resid_adj"] = af2full[m]["residual_species_R2_adj"]
            rec["AF2full_retention"] = af2full[m]["species_effect_retention_given_family_biophys"]
            rec["AF2full_species_p"] = af2full[m]["species_p_given_family_biophys"]
        # per-model departure of PDB from the matched-AFDB baseline
        sd = rec["AF2m_resid_raw_sd"] or np.nan
        z = (rec["PDB_resid_raw"] - rec["AF2m_resid_raw_mean"]) / sd
        rec["departure_z"] = z
        rec["departure_p"] = 2 * stats.norm.sf(abs(z))
        rows.append(rec)
    res = pd.DataFrame(rows)
    # BH-adjust the departure p across models
    order = res["departure_p"].rank(method="first")
    m_n = res["departure_p"].notna().sum()
    res["departure_p_BH"] = (res["departure_p"] * m_n / order).clip(upper=1.0)
    res.to_csv(RB / "data" / "vd_replication_stats.csv", index=False)

    # ---- (1) replication correlations ---------------------------------------
    names = res["model"].tolist()
    cors = []
    cors += corr_block(res["PDB_resid_raw"], res["AF2m_resid_raw_mean"], names,
                       "PDB vs AF2-matched (raw, same N=876)")
    have_full = res["AF2full_resid_adj"].notna()
    cors += corr_block(res.loc[have_full, "PDB_resid_adj"], res.loc[have_full, "AF2full_resid_adj"],
                       res.loc[have_full, "model"].tolist(), "PDB vs AF2-full (adjusted)")
    cors += corr_block(res.loc[have_full, "AF2m_resid_adj_mean"], res.loc[have_full, "AF2full_resid_adj"],
                       res.loc[have_full, "model"].tolist(), "AF2-matched vs AF2-full (adjusted)")
    cors += corr_block(res["PDB_retention"], res["AF2m_retention_mean"], names,
                       "PDB vs AF2-matched (retention)")
    cors = pd.DataFrame(cors)
    cors.to_csv(RB / "data" / "vd_replication_correlations.csv", index=False)

    # ---- (3) paired structure-vs-sequence test ------------------------------
    res["class"] = res["type"].map(lambda t: "sequence" if t == "sequence"
                                   else "structure/hybrid")
    paired = []
    for cls, g in res.groupby("class"):
        diff = (g["PDB_resid_raw"] - g["AF2m_resid_raw_mean"]).values
        try:
            w, p = stats.wilcoxon(diff)
        except ValueError:
            w, p = np.nan, np.nan
        paired.append(dict(model_class=cls, n=len(g), median_PDB_minus_AF2m=float(np.median(diff)),
                           wilcoxon_W=w, wilcoxon_p=p))
    paired = pd.DataFrame(paired)

    # ---- print summary -------------------------------------------------------
    pd.set_option("display.width", 200)
    print("=== Per-model residual species across inputs (+ departure of PDB from matched AF2) ===")
    show = ["model", "type", "PDB_resid_raw", "AF2m_resid_raw_mean", "AF2full_resid_adj",
            "departure_z", "departure_p_BH"]
    print(res[show].round(3).to_string(index=False))
    print("\n=== (1) Replication correlations across the model panel ===")
    print(cors.round(3).to_string(index=False))
    print("\n=== (3) Paired PDB-minus-AF2matched within model class (Wilcoxon) ===")
    print(paired.round(4).to_string(index=False))

    # ---- LaTeX table (correlations) -----------------------------------------
    def fnum(x): return "--" if pd.isna(x) else f"{x:.3f}"
    def fp(x):   return "--" if pd.isna(x) else ("$<$0.001" if x < 1e-3 else f"{x:.3f}")
    lines = []
    for _, r in cors.iterrows():
        lines.append(f"{r['comparison']} & {int(r['n_models'])} & "
                     f"{fnum(r['pearson_r'])} & {fp(r['pearson_p'])} & "
                     f"{fnum(r['spearman_rho'])} & {fp(r['spearman_p'])} \\\\")
    tex = ("% auto-generated by design/run_replication_stats.py\n"
           "\\begin{table}[ht]\\centering\\small\n"
           "\\caption{Replication of the per-model residual taxonomic bias across inputs: "
           "correlation of the per-model residual-species vector between the experimental-PDB "
           "cohort, the matched AFDB subsample, and the full AFDB dataset. Spearman (rank) "
           "tests whether the ordering of models by bias replicates; raw residuals are used "
           "where $N$ is matched, adjusted $R^2$ across different $N$.}"
           "\\label{tab:vd-replication}\n"
           "\\begin{tabular}{lrrrrr}\n\\toprule\n"
           "Comparison & $k$ & Pearson $r$ & $p$ & Spearman $\\rho$ & $p$ \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (RB / "tables" / "table_vd_replication.tex").write_text(tex)

    # ---- figure: scatter PDB vs AF2-matched (raw) + PDB vs AF2-full (adj) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, (xc, yc, xl, yl, ttl, sub) in zip(ax, [
        ("AF2m_resid_raw_mean", "PDB_resid_raw", "AF2-matched residual species (raw)",
         "PDB cohort residual species (raw)", "Same N (876): PDB vs matched AFDB",
         cors[cors.comparison.str.startswith("PDB vs AF2-matched (raw")]),
        ("AF2full_resid_adj", "PDB_resid_adj", "AF2-full residual species (adjusted)",
         "PDB cohort residual species (adjusted)", "Across N: PDB vs full AFDB",
         cors[cors.comparison.str.startswith("PDB vs AF2-full")])]):
        d = res.dropna(subset=[xc, yc])
        a.scatter(d[xc], d[yc], c=[TYPE_C.get(t, "#666") for t in d["type"]], s=70, zorder=3)
        lo = min(d[xc].min(), d[yc].min()); hi = max(d[xc].max(), d[yc].max())
        pad = (hi - lo) * 0.08
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5, label="y = x")
        a.set_xlim(lo - pad, hi + pad); a.set_ylim(lo - pad, hi + pad)   # fix limits before repel
        repel_labels(a, d[xc].values, d[yc].values, d["model"].tolist())
        sr = sub["spearman_rho"].iloc[0]; sp = sub["spearman_p"].iloc[0]
        pr = sub["pearson_r"].iloc[0]
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(ttl)
        a.text(0.04, 0.96, f"Spearman ρ={sr:.2f} (p={sp:.3f})\nPearson r={pr:.2f}",
               transform=a.transAxes, va="top", fontsize=10,
               bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        a.legend(loc="lower right", fontsize=8); a.grid(alpha=0.2)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=t)
               for t, c in TYPE_C.items() if t in set(res["type"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("Does the per-model residual taxonomic bias replicate across inputs?", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(RB / "figures" / "fig_vd_replication.png", dpi=150)
    fig.savefig(RB / "figures" / "fig_vd_replication.pdf"); plt.close(fig)

    print("\nWrote pdb_robustness/data/vd_replication_stats.csv, vd_replication_correlations.csv")
    print("Wrote pdb_robustness/tables/table_vd_replication.tex, figures/fig_vd_replication.{png,pdf}")


if __name__ == "__main__":
    main()
