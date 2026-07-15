"""Shared figure helpers. Presentation only - no analysis numbers are computed
here (they come from the analysis modules), so statistics stay independently
testable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from . import catalog  # noqa: E402

TYPE_C = catalog.TYPE_COLORS
_TYPE_ORDER = {"structure": 0, "hybrid": 1, "hybrid-ST": 2, "sequence": 3}
_LABEL_WRAP = {
    "ProteinMPNN": "Protein\nMPNN", "SolubleMPNN": "Soluble\nMPNN",
    "SolubleCaliby": "Soluble\nCaliby", "ESM3-struct": "ESM3\nstruct",
    "ESM3-seq": "ESM3\nseq", "CARP-640M": "CARP\n640M", "ESM2-15B": "ESM2\n15B",
}


def variance_decomposition_figures(res, out_dir, n_features: int = 14) -> list[Path]:
    """Draw the four variance-decomposition diagnostics as separate PDFs.

    The published SI figure (Fig S2) is the consolidated 3-panel view emitted by
    `variance_decomposition_combined_figure`; these four are the individual
    high-resolution panels (baseline/attenuation/unique/Simpson), ported verbatim
    from score_variance_decomposition.py (structure->hybrid->sequence ordering,
    same palettes)."""
    out_dir = Path(out_dir)
    FAM_C, SPC_C, BIO_C, SPC_LT = "#9E9E9E", "#EF6C00", "#1F9E9A", "#FFCC80"

    r = (res.assign(type_order=res["type"].map(_TYPE_ORDER))
            .sort_values(["type_order", "R2_Biophys"], ascending=[True, False])
            .reset_index(drop=True))
    x = np.arange(len(r))
    seq_start = int((r["type"] == "sequence").values.argmax()) if (r["type"] == "sequence").any() else len(r)
    htype = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_C.values()]
    ret = r["species_effect_retention_given_family_biophys"]
    corr = r["species_effect_corr_given_family_biophys"]
    cols = [TYPE_C[t] for t in r["type"]]
    model_labels = [_LABEL_WRAP.get(name, name) for name in r["model"]]

    def deco(a):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="y", alpha=0.2)
        if 0 < seq_start < len(r):
            a.axvline(seq_start - 0.5, color="k", ls="--", lw=0.9, alpha=0.45)
        a.set_xticks(x)
        a.set_xticklabels(model_labels, rotation=0, ha="center", fontsize=8)
        a.tick_params(axis="x", pad=7)

    def panel_baseline(a):
        w = 0.27
        if 0 < seq_start < len(r):
            a.text((seq_start - 1) / 2, 0.985, "structure-conditioned / hybrid",
                   transform=a.get_xaxis_transform(), ha="center", va="top",
                   fontsize=8.5, color="#555555")
            a.text((seq_start + len(r) - 1) / 2, 0.985, "sequence-only",
                   transform=a.get_xaxis_transform(), ha="center", va="top",
                   fontsize=8.5, color="#555555")
        a.bar(x - w, r["R2_Family"], w, color=FAM_C, label="protein family")
        a.bar(x, r["R2_Biophys"], w, color=BIO_C, label=f"biophysics ({n_features} features)")
        a.bar(x + w, r["R2_Species"], w, color=SPC_C, label="species")
        a.set_ylabel("R²  (factor alone)")
        a.set_ylim(0, r["R2_Family"].max() * 1.18)
        a.set_title("Score variance explained by each factor alone", fontsize=12, pad=26)
        a.legend(ncol=3, fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))

    def panel_attenuation(a):
        a.bar(x - 0.2, r["R2_Species"], 0.4, color=SPC_LT, label="species alone (before adjustment)")
        a.bar(x + 0.2, r["dSpecies_given_family_biophys"], 0.4, color=SPC_C,
              label="species after family + biophysics")
        for i, (s0, att) in enumerate(zip(r["R2_Species"], r["species_attenuation"])):
            a.text(i, s0 + 0.006, f"−{att*100:.0f}%", ha="center", va="bottom",
                   fontsize=7.5, color="#555555")
        a.set_ylabel("R²:  species")
        a.set_ylim(0, r["R2_Species"].max() * 1.24)
        a.set_title("Species signal attenuation after family + biophysics", fontsize=12, pad=26)
        a.legend(fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    def panel_unique(a):
        a.bar(x - 0.2, r["dBiophys_given_family_species"], 0.4, color=BIO_C,
              label="biophysics | family + species")
        a.bar(x + 0.2, r["dSpecies_given_family_biophys"], 0.4, color=SPC_C,
              label="species | family + biophysics")
        a.set_ylabel("ΔR²  (unique)")
        a.set_ylim(0, max(r["dSpecies_given_family_biophys"].max(),
                          r["dBiophys_given_family_species"].max()) * 1.18)
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
        a.legend(hleg, list(TYPE_C.keys()) + ["correlation"], fontsize=8.5,
                 frameon=False, loc="upper left", ncol=2)

    panels = [
        ("fig_vd_baseline.pdf", panel_baseline),
        ("fig_vd_attenuation.pdf", panel_attenuation),
        ("fig_vd_unique.pdf", panel_unique),
        ("fig_vd_simpson.pdf", panel_simpson),
    ]
    written = []
    for fname, drawer in panels:
        fig, a = plt.subplots(figsize=(12, 5.2))
        drawer(a)
        deco(a)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
        plt.close(fig)
        written.append(out_dir / fname)
    written += variance_decomposition_combined_figure(res, out_dir, n_features=n_features)
    return written


def variance_decomposition_combined_figure(res, out_dir, n_features: int = 14) -> list[Path]:
    """Single 3-panel variance-decomposition figure (SI Fig S2).

    Stacks the species-attenuation, unique-contribution and within-family
    (Simpson's) diagnostics into one print-ready figure, the consolidated view
    that replaced the four separate SI panels. Same data, ordering and palette
    as `variance_decomposition_figures`."""
    out_dir = Path(out_dir)
    SPC_C, BIO_C, SPC_LT = "#EF6C00", "#1F9E9A", "#FFCC80"
    r = (res.assign(type_order=res["type"].map(_TYPE_ORDER))
            .sort_values(["type_order", "R2_Biophys"], ascending=[True, False])
            .reset_index(drop=True))
    x = np.arange(len(r))
    seq_start = int((r["type"] == "sequence").values.argmax()) if (r["type"] == "sequence").any() else len(r)
    labels = [_LABEL_WRAP.get(name, name) for name in r["model"]]

    def deco(a):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="y", alpha=0.2)
        if 0 < seq_start < len(r):
            a.axvline(seq_start - 0.5, color="k", ls="--", lw=0.9, alpha=0.45)
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=8)
        a.tick_params(axis="x", pad=6)

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 13.5))

    a1.bar(x - 0.2, r["R2_Species"], 0.4, color=SPC_LT, label="species alone (before adjustment)")
    a1.bar(x + 0.2, r["dSpecies_given_family_biophys"], 0.4, color=SPC_C,
           label="species after family + biophysics")
    for i, (s0, att) in enumerate(zip(r["R2_Species"], r["species_attenuation"])):
        a1.text(i, s0 + 0.006, f"−{att*100:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#555555")
    a1.set_ylabel("R²:  species")
    a1.set_ylim(0, r["R2_Species"].max() * 1.24)
    a1.set_title("a  Species-signal attenuation after family + biophysics", fontsize=12, loc="left")
    a1.legend(fontsize=9, frameon=False, loc="upper right")

    a2.bar(x - 0.2, r["dBiophys_given_family_species"], 0.4, color=BIO_C, label="biophysics | family + species")
    a2.bar(x + 0.2, r["dSpecies_given_family_biophys"], 0.4, color=SPC_C, label="species | family + biophysics")
    a2.set_ylabel("ΔR²  (unique)")
    a2.set_ylim(0, max(r["dSpecies_given_family_biophys"].max(),
                       r["dBiophys_given_family_species"].max()) * 1.18)
    a2.set_title("b  Unique contribution beyond the other two factors (all p < 0.001)", fontsize=12, loc="left")
    a2.legend(fontsize=9, frameon=False, loc="upper left")

    ret = r["species_effect_retention_given_family_biophys"]
    corr = r["species_effect_corr_given_family_biophys"]
    a3.bar(x, ret, color=[TYPE_C[t] for t in r["type"]])
    a3.plot(x, corr, "ko", ms=5)
    a3.axhline(1.0, color="k", ls=":", lw=0.8)
    for i, v in enumerate(ret):
        a3.text(i, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    a3.set_ylabel("per-species effect:\nretention (bars), corr (dots)")
    a3.set_ylim(0, 1.18)
    a3.set_title("c  Within-family consistency of per-species effects (Simpson's check)", fontsize=12, loc="left")
    hleg = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_C.values()] + \
        [plt.Line2D([0], [0], marker="o", color="k", ls="", ms=5)]
    a3.legend(hleg, list(TYPE_C.keys()) + ["correlation"], fontsize=8.5, frameon=False, loc="upper left", ncol=2)

    for a in (a1, a2, a3):
        deco(a)
    fig.tight_layout(h_pad=2.2)
    written = []
    for ext, dpi in (("pdf", 600), ("png", 200)):
        p = out_dir / f"fig_variance_decomposition.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written
