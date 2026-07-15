"""
WT->design predicted-melting-temperature (DeepStabP) shift, per model.

Input: design/Design_WT_TM.csv  (protein_id, Tm)
  designs: <uniprot>__<model>__s<idx>     WTs: <uniprot>__WT

Per (model, protein): mean design Tm over the 8 replicates, paired against that
protein's WT Tm -> Delta Tm. Across the 25 proteins: paired Cohen's d_z, Wilcoxon
signed-rank, BH-FDR across models. Also per domain, and distributions.
"""
import sys, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TM = REPO / "design" / "Design_WT_TM.csv"
INP = REPO / "design" / "design_input_proteins.csv"
OUT = REPO / "design" / "outputs"; OUT.mkdir(exist_ok=True)


def parse_id(pid):
    if pid.endswith("__WT"):
        return pid[:-4], "WT", -1
    m = re.match(r"(.+)__(.+)__s(\d+)$", pid)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (pid, "?", -1)


def main():
    t = pd.read_csv(TM)
    t[["uniprot_id", "model", "sample_idx"]] = pd.DataFrame(
        [parse_id(p) for p in t.protein_id], index=t.index)
    meta = pd.read_csv(INP)[["uniprot_id", "domain", "rank_class", "species"]]
    t = t.merge(meta, on="uniprot_id", how="left")

    wt = t[t.model == "WT"].set_index("uniprot_id")["Tm"]
    des = t[t.model != "WT"].copy()
    des["wt_Tm"] = des.uniprot_id.map(wt)
    des["dTm"] = des.Tm - des.wt_Tm

    # per-protein mean ΔTm, then per-model stats across proteins
    pp = des.groupby(["model", "uniprot_id", "domain"]).agg(
        dTm=("dTm", "mean"), design_Tm=("Tm", "mean"), wt_Tm=("wt_Tm", "first")).reset_index()
    pp.to_csv(OUT / "tm_shift_per_protein.csv", index=False)

    rows = []
    for model, g in pp.groupby("model"):
        x = g.dTm.values
        dz = x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan
        p = stats.wilcoxon(x).pvalue if len(x) >= 3 else np.nan
        rows.append(dict(model=model, mean_dTm=x.mean(), median_dTm=np.median(x),
                         sd_dTm=x.std(ddof=1), cohens_dz=dz, wilcoxon_p=p, n=len(x)))
    res = pd.DataFrame(rows)
    from scipy.stats import false_discovery_control
    res["p_fdr"] = false_discovery_control(res.wilcoxon_p.fillna(1))
    res = res.sort_values("mean_dTm", ascending=False)
    res.to_csv(OUT / "tm_shift_by_model.csv", index=False)
    print("=== WT→design ΔTm (°C) per model (DeepStabP) ===")
    print(res[["model", "mean_dTm", "median_dTm", "cohens_dz", "p_fdr", "n"]].round(2).to_string(index=False))

    # per model x domain
    dom = pp.groupby(["model", "domain"]).dTm.agg(["mean", "median", "count"]).round(2).reset_index()
    dom.to_csv(OUT / "tm_shift_by_model_domain.csv", index=False)
    print("\n=== mean ΔTm by model × domain ===")
    print(dom.pivot(index="model", columns="domain", values="mean").round(1).to_string())

    # plot: ΔTm distribution per model, coloured by domain
    fig, ax = plt.subplots(figsize=(11, 6))
    order = res.model.tolist()
    dom_cols = {"Archaea": "#F57C00", "Bacteria": "#1976D2", "Eukaryota": "#388E3C"}
    for i, m in enumerate(order):
        g = pp[pp.model == m]
        # jittered points coloured by domain + a box
        ax.boxplot(g.dTm, positions=[i], widths=0.6, showfliers=False,
                   medianprops=dict(color="black"))
        for dmn, gg in g.groupby("domain"):
            ax.scatter(np.full(len(gg), i) + np.random.uniform(-0.18, 0.18, len(gg)),
                       gg.dTm, c=dom_cols.get(dmn, "grey"), s=22, alpha=0.8, label=dmn)
    ax.axhline(0, color="red", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("ΔTm (design − WT), °C"); ax.set_title("Predicted thermostability shift per model (per-protein means)")
    h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), title="domain", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "tm_shift_distributions.png", dpi=150)
    print("\nwrote tm_shift_distributions.png, tm_shift_by_model.csv, tm_shift_per_protein.csv")


if __name__ == "__main__":
    main()
