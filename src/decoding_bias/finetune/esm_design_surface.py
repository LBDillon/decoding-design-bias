"""Surface composition of the ESM designs (Phase 2), mapped onto the template
backbone (RSA>=0.25), compared WT -> design across base / AlkSec / Neu.
Also flags low-complexity collapse (a known ESM in-filling failure mode).

  python esm_design_surface.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "design"))
from surface_features_alkaline import per_residue_rsa, comp, MAXASA, RSA_CUT

GEN = ROOT / "outputs" / "esm35m_continual_pretraining" / "generation"


def surface_letters(seq, surf_idx):
    return [seq[i] for i in surf_idx if i < len(seq) and seq[i] in MAXASA]


def complexity(seq):
    """Shannon entropy (bits) of the AA composition - low = collapsed."""
    from collections import Counter
    n = len(seq)
    p = np.array([v / n for v in Counter(seq).values()])
    return float(-(p * np.log2(p)).sum())


def main():
    src = GEN / "esm_designs_local_surface.csv"
    if not src.exists():
        src = GEN / "esm_designs_local.csv"
    print(f"designs: {src.name}")
    designs = pd.read_csv(src)
    inp = pd.read_csv(ROOT / "design" / "design_input_proteins.csv").set_index("uniprot_id")
    rows = []
    for uid, g in designs.groupby("uniprot_id"):
        path = inp.loc[uid, "structure_pdb_v6"]
        letters, rsa = per_residue_rsa(path)
        surf_idx = [i for i, r in enumerate(rsa) if not np.isnan(r) and r >= RSA_CUT]
        wt = inp.loc[uid, "wt_sequence"]
        wt_net = comp(surface_letters(wt, surf_idx)).get("net_KR_DE", np.nan)
        for model, gm in g.groupby("model"):
            nets = [comp(surface_letters(s, surf_idx)).get("net_KR_DE", np.nan) for s in gm.sequence]
            ents = [complexity(s) for s in gm.sequence]
            rows.append(dict(uniprot_id=uid, model=model,
                             surf_net=np.nanmean(nets), surf_net_shift=np.nanmean(nets) - wt_net,
                             mean_entropy=np.mean(ents), wt_surf_net=wt_net))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="uniprot_id", columns="model", values="surf_net")
    ent = df.pivot_table(index="uniprot_id", columns="model", values="mean_entropy")
    print("=== surface net charge (K+R-D-E)/n_surf of designs ===")
    print(piv.round(3).to_string())
    print("\n=== WT surface net charge ===")
    print(df.groupby("uniprot_id").wt_surf_net.first().round(3).to_string())
    print("\n=== design sequence entropy (bits; WT-like ~4; <2 = collapsed) ===")
    print(ent.round(2).to_string())
    df.to_csv(GEN / "esm_design_surface_summary.csv", index=False)
    print("\nwrote esm_design_surface_summary.csv")


if __name__ == "__main__":
    main()
