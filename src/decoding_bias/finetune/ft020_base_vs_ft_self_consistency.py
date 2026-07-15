"""
Native v_48_020 FT-vs-base self-consistency test.

The 200 ProteinMPNN (v_48_020 base) designs were folded in the main design run
(design/arc_downloads/rank001_flat). Score them vs the AFDB input backbone with the
identical tm_align protocol used for the FT-020 designs, then run the paired
fine-tuned-vs-base test (by template) so the base comparison is native to v_48_020
rather than borrowed from the 002 arm.

  python paper_code/09_model_diagnostics/ft020_base_vs_ft_self_consistency.py
"""
import os, glob, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "design"))
import design_common as dc

BASE_DIR = os.path.join(ROOT, "design", "arc_downloads", "rank001_flat")
OUT = os.path.join(HERE, "outputs")
FT_CSV = os.path.join(OUT, "ft020_self_consistency_vs_afdb.csv")

ref_path = dict(zip(dc.load_inputs().uniprot_id, dc.load_inputs().structure_path))


def ca(path):
    chain = next(get_structure(path).get_chains())
    coords, seq = get_residue_data(chain)
    return np.asarray(coords, float), seq


def mean_plddt(path):
    vals = [float(l[60:66]) for l in open(path)
            if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return float(np.mean(vals)) if vals else np.nan


def p_label(p):
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


ref_ca = {u: ca(p) for u, p in ref_path.items()}

# --- score the 200 base ProteinMPNN (v_48_020) designs vs AFDB ---
rows = []
for p in glob.glob(os.path.join(BASE_DIR, "*__ProteinMPNN__*rank_001*.pdb")):
    fid = os.path.basename(p).split("_unrelaxed_rank")[0]
    m = re.match(r"(.+?)__(.+?)__s(\d+)$", fid)
    if not m or m.group(1) not in ref_ca:
        continue
    uni, s = m.group(1), int(m.group(3))
    dc_, ds = ca(p)
    rc, rs = ref_ca[uni]
    r = tm_align(dc_, rc, ds, rs)
    rows.append(dict(uniprot_id=uni, model="ProteinMPNN_v020(base)", sample_idx=s,
                     scTM=r.tm_norm_chain2, scRMSD=r.rmsd, pLDDT=mean_plddt(p)))
base = pd.DataFrame(rows)
print(f"scored {len(base)} base ProteinMPNN(v_48_020) designs; templates={base.uniprot_id.nunique()}")

# --- combine with the FT-020 + WT-control results, save ---
ft = pd.read_csv(FT_CSV)
comb = pd.concat([ft, base], ignore_index=True)
comb.to_csv(os.path.join(OUT, "ft020_self_consistency_vs_afdb_with_base.csv"), index=False)

print("\n=== per-model self-consistency vs AFDB backbone (v_48_020) ===")
print(comb.groupby("model")[["scTM", "scRMSD", "pLDDT"]].agg(["mean", "median"]).round(3))

# --- paired FT-vs-base, by template (mean over designs per template) ---
b = base.groupby("uniprot_id")[["scTM", "scRMSD", "pLDDT"]].mean()
print("\n=== fine-tuned vs base (v_48_020), paired by template, Wilcoxon signed-rank ===")
tests = {}
for ftm in ["AlkSecMPNN_020", "AcidSecMPNN_020"]:
    f = comb[comb.model == ftm].groupby("uniprot_id")[["scTM", "scRMSD", "pLDDT"]].mean()
    c = b.index.intersection(f.index)
    for met in ["scTM", "scRMSD", "pLDDT"]:
        d = (f.loc[c, met] - b.loc[c, met])
        try:
            pv = wilcoxon(f.loc[c, met], b.loc[c, met]).pvalue
        except ValueError:
            pv = float("nan")
        tests[(ftm, met)] = (d.mean(), pv)
        print(f"  {ftm:18s} {met:6s}  FT={f.loc[c,met].mean():.3f}  base={b.loc[c,met].mean():.3f}"
              f"  delta={d.mean():+.3f}  p={pv:.3f}")

plot = comb.groupby(["model", "uniprot_id"], as_index=False)[["scTM", "scRMSD", "pLDDT"]].mean()
order = ["ProteinMPNN_v020(base)", "AlkSecMPNN_020", "AcidSecMPNN_020", "WT_singleseq(control)"]
labels = ["ProteinMPNN\nbase", "AlkSecMPNN", "AcidSecMPNN", "WT\nsingle-seq"]
colors = ["#7f8c8d", "#2c7fb8", "#d95f02", "#bdbdbd"]
metrics = [("scTM", "Self-consistency TM", "higher is better"),
           ("scRMSD", "Cα-RMSD to input (Å)", "lower is better"),
           ("pLDDT", "Refold pLDDT", "higher is better")]

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
rng = np.random.default_rng(3)
for ax, (met, title, subtitle) in zip(axes, metrics):
    vals = [plot.loc[plot.model == m, met].dropna().values for m in order]
    bp = ax.boxplot(vals, positions=np.arange(len(order)), widths=0.55,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for part in ["whiskers", "caps", "medians"]:
        for artist in bp[part]:
            artist.set_color("#333333")
    for i, v in enumerate(vals):
        jitter = rng.normal(0, 0.035, size=len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=14, color=colors[i],
                   alpha=0.75, linewidths=0)

    ymax = max(max(v) for v in vals if len(v))
    ymin = min(min(v) for v in vals if len(v))
    pad = (ymax - ymin) * 0.18 if ymax > ymin else 1
    ax.set_ylim(ymin - pad * 0.25, ymax + pad)
    for i, ftm in [(1, "AlkSecMPNN_020"), (2, "AcidSecMPNN_020")]:
        delta, p = tests[(ftm, met)]
        ax.text(i, ymax + pad * 0.10,
                f"Δ={delta:+.2f}\n{p_label(p)}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("v_48_020 fine-tuned design refolds vs AFDB input backbone", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ft020_self_consistency_vs_afdb_with_base.png"),
            dpi=150, bbox_inches="tight")

print("\nSaved ft020_self_consistency_vs_afdb_with_base.csv")
print("Saved ft020_self_consistency_vs_afdb_with_base.png")
