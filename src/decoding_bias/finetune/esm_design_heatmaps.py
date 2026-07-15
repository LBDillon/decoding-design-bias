"""Phase-1 analysis: does fine-tuning change ESM2-35M's per-position amino-acid
preferences, and does any shift concentrate at the surface? (Reviewer R1.4.)

Reads the per-template probability maps produced on Colab
(outputs/esm35m_continual_pretraining/generation/<ID>_probs.npz, each with `seq`,
`aa_order`, `base`, `AlkSecESM35M`). For each template it writes:
  * a Delta(FT-base) amino-acid heatmap over positions (acidic/basic rows flagged),
    with the acidic-propensity track (base vs FT) and surface positions marked;
  * a summary row: mean Delta acidic-propensity overall / at surface / in core.

  python esm_design_heatmaps.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))                       # esm_generation
sys.path.insert(0, str(ROOT / "design"))            # surface_features_alkaline
from esm_generation import acidic_propensity, CANONICAL_AA, ACIDIC, BASIC

GEN_DIR = ROOT / "outputs" / "esm35m_continual_pretraining" / "generation"
FT_KEY = "AlkSecESM35M"      # the steered fine-tune
CTRL_KEY = "NeuSecESM35M"    # matched neutralophile-control fine-tune (specificity)
RSA_CUT = 0.25


def structure_paths():
    d = pd.read_csv(ROOT / "design" / "design_input_proteins.csv")
    return dict(zip(d.uniprot_id, d.structure_pdb_v6))


def surface_mask(uid, seq, paths):
    """Boolean per-position surface mask (RSA>=0.25); None if unavailable."""
    p = paths.get(uid)
    if not p or not Path(p).exists():
        return None
    try:
        from surface_features_alkaline import per_residue_rsa
        _, rsa = per_residue_rsa(p)
    except Exception as e:
        print(f"  {uid}: surface RSA unavailable ({e})")
        return None
    if len(rsa) != len(seq):
        print(f"  {uid}: RSA length {len(rsa)} != seq {len(seq)}; skipping surface overlay")
        return None
    return np.nan_to_num(rsa, nan=0.0) >= RSA_CUT


def plot_template(uid, npz, paths, out_dir):
    seq = str(npz["seq"])
    aa = "".join(npz["aa_order"].tolist()) if "aa_order" in npz else CANONICAL_AA
    base, ft = npz["base"], npz[FT_KEY]
    delta = ft - base                                  # (L, 20)
    prop_base = acidic_propensity(base, aa)
    prop_ft = acidic_propensity(ft, aa)
    dprop = prop_ft - prop_base
    has_ctrl = CTRL_KEY in npz.files
    if has_ctrl:
        prop_ctrl = acidic_propensity(npz[CTRL_KEY], aa)
        dprop_ctrl = prop_ctrl - prop_base
    surf = surface_mask(uid, seq, paths)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(7, len(seq) * 0.12), 6.5),
                                   gridspec_kw={"height_ratios": [3, 1.4]}, sharex=True)
    vmax = np.abs(delta).max() or 1e-6
    im = ax1.imshow(delta.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_yticks(range(len(aa))); ax1.set_yticklabels(list(aa), fontsize=7)
    for i, a in enumerate(aa):
        if a in ACIDIC: ax1.get_yticklabels()[i].set_color("#B2182B")
        if a in BASIC:  ax1.get_yticklabels()[i].set_color("#2166AC")
    ax1.set_ylabel("amino acid"); ax1.set_title(
        f"{uid}  ({len(seq)} aa)   FT - base per-position probability  (red=FT prefers more)")
    fig.colorbar(im, ax=ax1, fraction=0.02, pad=0.01, label="Δ prob")

    ax2.axhline(0, color="grey", lw=0.6)
    ax2.plot(prop_base, color="grey", lw=1, label="base")
    ax2.plot(prop_ft, color="#B2182B", lw=1, label="AlkSec FT")
    if has_ctrl:
        ax2.plot(prop_ctrl, color="#2166AC", lw=1, ls="--", label="NeuSec ctrl")
    ax2.set_ylabel("acidic\npropensity", fontsize=8)
    ax2.set_xlabel("position"); ax2.legend(fontsize=7, loc="upper right")
    if surf is not None:
        for x in np.where(surf)[0]:
            ax2.axvspan(x - 0.5, x + 0.5, color="#f4d03f", alpha=0.18, lw=0)
        ax2.text(0.01, 0.02, "shaded = surface (RSA≥0.25)", transform=ax2.transAxes, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_esm_heatmap_{uid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    row = {"uniprot_id": uid, "length": len(seq),
           "alk_dprop_all": float(np.mean(dprop))}
    if has_ctrl:
        row["neu_dprop_all"] = float(np.mean(dprop_ctrl))
    if surf is not None and surf.any():
        core = ~surf
        row["alk_dprop_surface"] = float(np.mean(dprop[surf]))
        row["alk_dprop_core"] = float(np.mean(dprop[core])) if core.any() else np.nan
        if has_ctrl:
            row["neu_dprop_surface"] = float(np.mean(dprop_ctrl[surf]))
            # cohort-specific surface shift = steered minus control
            row["surface_specificity"] = row["alk_dprop_surface"] - row["neu_dprop_surface"]
        row["n_surface"] = int(surf.sum())
    return row


def main():
    out_dir = GEN_DIR
    npz_files = sorted(GEN_DIR.glob("*_probs.npz"))
    if not npz_files:
        print(f"No *_probs.npz in {GEN_DIR}. Run the Phase-1 Colab notebook and unzip here.")
        return
    paths = structure_paths()
    rows = []
    for f in npz_files:
        uid = f.name.replace("_probs.npz", "")
        with np.load(f, allow_pickle=True) as npz:
            rows.append(plot_template(uid, npz, paths, out_dir))
        print(f"  wrote fig_esm_heatmap_{uid}.png")
    summ = pd.DataFrame(rows)
    summ.to_csv(GEN_DIR / "esm_design_heatmap_summary.csv", index=False)
    print("\n=== Δ acidic-propensity (FT - base): >0 means FT prefers MORE acidic ===")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
