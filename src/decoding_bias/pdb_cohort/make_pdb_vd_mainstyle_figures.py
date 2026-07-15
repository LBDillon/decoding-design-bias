"""
Reproduce the four MAIN-cohort variance-decomposition panels
(baseline / attenuation / unique / simpson) for the experimental-PDB cohort,
in the EXACT style of paper_code/03_variance_decomposition/score_variance_decomposition.py.

Only the data source changes: instead of recomputing, we read the already-computed
per-model PDB-cohort decomposition (design/outputs/independent_cohort/
cohort_score_variance_decomposition.csv). The plotting block below is copied
verbatim from score_variance_decomposition.py so the figures match the main set.

Fine-tuned arms (type "structure(FT)") are dropped so the panel shows the same
base-model set as the main figure. A discreet sup-title marks the cohort so the
PDB figures cannot be confused with the main ones.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[0] / "design" / "outputs" / "independent_cohort" / "cohort_score_variance_decomposition.csv"
OUT = HERE / "figures"
SUPTITLE = "Experimental-PDB cohort (n = 876; 13 models, TriFlow excluded)"

res = pd.read_csv(CSV)
# base models only (drop the two fine-tuned arms), matching the main figure's panel
res = res[~res["type"].astype(str).str.contains("FT")].reset_index(drop=True)

# ===== plotting block copied verbatim from score_variance_decomposition.py =====
type_order = {"structure": 0, "hybrid": 1, "hybrid-ST": 2, "sequence": 3}
r = (
    res.assign(type_order=res["type"].map(type_order))
    .sort_values(["type_order", "R2_Biophys"], ascending=[True, False])
    .reset_index(drop=True)
)
x = np.arange(len(r))
TYPE_C = {"structure": "#2E7D32", "hybrid": "#F9A825", "hybrid-ST": "#C0CA33", "sequence": "#00838F"}
cols = [TYPE_C[t] for t in r["type"]]
FAM_C, SPC_C, BIO_C, SPC_LT = "#9E9E9E", "#EF6C00", "#1F9E9A", "#FFCC80"
seq_start = int((r["type"] == "sequence").values.argmax()) if (r["type"] == "sequence").any() else len(r)
htype = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_C.values()]
ret = r["species_effect_retention_given_family_biophys"]
corr = r["species_effect_corr_given_family_biophys"]


def deco(a):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
    if 0 < seq_start < len(r):
        a.axvline(seq_start - 0.5, color="k", ls="--", lw=0.9, alpha=0.45)
    a.set_xticks(x); a.set_xticklabels(r["model"], rotation=40, ha="right")


def panel_baseline(a):
    w = 0.27
    if 0 < seq_start < len(r):
        a.text((seq_start - 1) / 2, 0.985, "structure-conditioned / hybrid",
               transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color="#555555")
        a.text((seq_start + len(r) - 1) / 2, 0.985, "sequence-only",
               transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color="#555555")
    a.bar(x - w, r["R2_Family"], w, color=FAM_C, label="protein family")
    a.bar(x,     r["R2_Biophys"], w, color=BIO_C, label="biophysics (14 features)")
    a.bar(x + w, r["R2_Species"], w, color=SPC_C, label="species")
    a.set_ylabel("R²  (factor alone)")
    a.set_ylim(0, r["R2_Family"].max() * 1.18)
    a.set_title("Score variance explained by each factor alone", fontsize=12, pad=26)
    a.legend(ncol=3, fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))


def panel_attenuation(a):
    a.bar(x - 0.2, r["R2_Species"], 0.4, color=SPC_LT, label="species alone (before adjustment)")
    a.bar(x + 0.2, r["dSpecies_given_family_biophys"], 0.4, color=SPC_C, label="species after family + biophysics")
    for i, (s0, att) in enumerate(zip(r["R2_Species"], r["species_attenuation"])):
        a.text(i, s0 + 0.006, f"−{att*100:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#555555")
    a.set_ylabel("R²:  species")
    a.set_ylim(0, r["R2_Species"].max() * 1.24)
    a.set_title("Species signal attenuation after family + biophysics", fontsize=12, pad=26)
    a.legend(fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)


def panel_unique(a):
    a.bar(x - 0.2, r["dBiophys_given_family_species"], 0.4, color=BIO_C, label="biophysics | family + species")
    a.bar(x + 0.2, r["dSpecies_given_family_biophys"], 0.4, color=SPC_C, label="species | family + biophysics")
    a.set_ylabel("ΔR²  (unique)")
    a.set_ylim(0, max(r["dSpecies_given_family_biophys"].max(), r["dBiophys_given_family_species"].max()) * 1.18)
    a.set_title("Unique contribution of each factor beyond the other two (all p < 0.001)", fontsize=12)
    a.legend(fontsize=9, frameon=False, loc="upper left")


def panel_simpson(a):
    a.bar(x, ret, color=cols)
    a.plot(x, corr, "ko", ms=5)
    a.axhline(1.0, color="k", ls=":", lw=0.8)
    for i, v in enumerate(ret):
        a.text(i, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=7, color="black",
               path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    a.set_ylabel("per-species effect:\nretention (bars), corr (dots)")
    a.set_ylim(0, 1.18)
    a.set_title("Within-family consistency of per-species effects (Simpson's check)", fontsize=12)
    hleg = htype + [plt.Line2D([0], [0], marker="o", color="k", ls="", ms=5)]
    a.legend(hleg, list(TYPE_C.keys()) + ["correlation"], fontsize=8.5, frameon=False, loc="upper left", ncol=2)


PANELS = [
    ("fig_pdb_vd_baseline.png", panel_baseline),
    ("fig_pdb_vd_attenuation.png", panel_attenuation),
    ("fig_pdb_vd_unique.png", panel_unique),
    ("fig_pdb_vd_simpson.png", panel_simpson),
]
OUT.mkdir(parents=True, exist_ok=True)
for fname, drawer in PANELS:
    fig, a = plt.subplots(figsize=(12, 5.2))
    drawer(a); deco(a)
    fig.suptitle(SUPTITLE, fontsize=9, color="#777777", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150, bbox_inches="tight"); plt.close(fig)

print("models plotted:", list(r["model"]))
print("wrote:", ", ".join(str(OUT / n) for n, _ in PANELS))
