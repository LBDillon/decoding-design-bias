"""
Self-consistency of the v_48_020 fine-tuned designs vs their AFDB input backbone.

Mirrors the 002 vs-AFDB analysis (ft_self_consistency_vs_afdb.csv): each ColabFold
single-sequence refold (rank_001) is TM-aligned to the AFDB v6 backbone it was
designed on; scTM is normalised by the AFDB reference (tm_norm_chain2). The 25
wild-type single-sequence refolds give the same-reference control.

  python paper_code/09_model_diagnostics/ft020_self_consistency_vs_afdb.py
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

AF2_DIR = os.path.join(ROOT, "design", "outputs", "colabfold_out_ft020")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

ref_path = dict(zip(dc.load_inputs().uniprot_id, dc.load_inputs().structure_path))


def ca(path):
    chain = next(get_structure(path).get_chains())
    coords, seq = get_residue_data(chain)
    return np.asarray(coords, float), seq


def mean_plddt(path):
    vals = [float(l[60:66]) for l in open(path)
            if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return float(np.mean(vals)) if vals else np.nan


def fold_id(path):
    return os.path.basename(path).split("_unrelaxed_rank")[0]


pdbs = glob.glob(os.path.join(AF2_DIR, "*rank_001*.pdb"))
print(f"{len(pdbs)} rank-1 structures in {AF2_DIR}")

ref_ca = {u: ca(p) for u, p in ref_path.items()}
rows = []
for p in pdbs:
    fid = fold_id(p)
    if fid.endswith("__WT"):
        uni, model, s = fid[:-4], "WT_singleseq(control)", -1
    else:
        m = re.match(r"(.+?)__(.+?)__s(\d+)$", fid)
        if not m:
            continue
        uni, model, s = m.group(1), m.group(2), int(m.group(3))
    if uni not in ref_ca:
        continue
    dcoords, dseq = ca(p)
    rcoords, rseq = ref_ca[uni]
    r = tm_align(dcoords, rcoords, dseq, rseq)     # design vs AFDB (chain2 = reference)
    rows.append(dict(uniprot_id=uni, model=model, sample_idx=s,
                     scTM=r.tm_norm_chain2, scRMSD=r.rmsd, pLDDT=mean_plddt(p)))

sc = pd.DataFrame(rows)
sc.to_csv(os.path.join(OUT, "ft020_self_consistency_vs_afdb.csv"), index=False)
print(f"\n{len(sc)} structures scored ->", os.path.join(OUT, "ft020_self_consistency_vs_afdb.csv"))

print("\n=== per-model self-consistency vs AFDB backbone ===")
print(sc.groupby("model")[["scTM", "scRMSD", "pLDDT"]].agg(["mean", "median"]).round(3))

# FT vs WT-control, paired by template (mean over designs per template)
ctrl = sc[sc.model == "WT_singleseq(control)"].set_index("uniprot_id")["scTM"]
print("\n=== FT designs vs WT single-seq control (paired by template, Wilcoxon) ===")
for ft in ["AlkSecMPNN_020", "AcidSecMPNN_020"]:
    f = sc[sc.model == ft].groupby("uniprot_id")["scTM"].mean()
    c = f.index.intersection(ctrl.index)
    try:
        pv = wilcoxon(f.loc[c], ctrl.loc[c]).pvalue
    except ValueError:
        pv = float("nan")
    print(f"  {ft:20s} scTM={f.mean():.3f}  control={ctrl.loc[c].mean():.3f}"
          f"  delta={(f.loc[c]-ctrl.loc[c]).mean():+.3f}  p={pv:.2e}")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
order = ["AlkSecMPNN_020", "AcidSecMPNN_020", "WT_singleseq(control)"]
sc["model"] = pd.Categorical(sc["model"], order, ordered=True)
for a, met, lab in zip(ax, ["scTM", "scRMSD", "pLDDT"],
                       ["self-consistency TM (vs AFDB)", "self-consistency RMSD (A)", "design pLDDT"]):
    sc.boxplot(column=met, by="model", ax=a, grid=False)
    a.set_title(lab); a.set_xlabel("")
    a.set_xticklabels([t.get_text().replace("_singleseq(control)", "").replace("MPNN_020", "MPNN")
                       for t in a.get_xticklabels()], rotation=15)
plt.suptitle("v_48_020 fine-tuned design self-consistency vs AFDB backbone (25 templates x 8 designs)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ft020_self_consistency_vs_afdb.png"), dpi=150, bbox_inches="tight")
print("\nSaved ft020_self_consistency_vs_afdb.png")
