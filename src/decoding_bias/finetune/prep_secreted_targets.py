"""Pick N secreted held-out neutralophile test backbones (the targets AlkSecMPNN was
validated on) for the matched surface-only ProteinMPNN-vs-ESM redesign comparison.

Writes (to outputs/esm35m_continual_pretraining/generation/):
  * secreted_targets.csv          name, seq, structure_path, n_res, n_surface
  * secreted_surface_positions.json   {name: {surface: [...], len: L}}

QC: keep only backbones whose structure length matches the sequence, so surface
positions align. Run in an env with biotite (base): python prep_secreted_targets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "design"))
from surface_features_alkaline import per_residue_rsa, RSA_CUT

TEST_JSONL = ROOT / "finetune" / "data" / "alkaliphile_parsed_test.jsonl"
STRUCT_DIR = ROOT / "finetune" / "data" / "structures_alkaliphile" / "controls"
OUT = ROOT / "outputs" / "esm35m_continual_pretraining" / "generation"
N_TARGETS = 10
LEN_MIN, LEN_MAX = 50, 300   # keep it quick and avoid very long backbones
MAX_SURF_FRAC = 0.85         # exclude disordered/extended structures (near-all surface)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(TEST_JSONL)]
    ctrl = sorted([r for r in rows if r["role"] == "control"], key=lambda r: r["name"])

    targets, surf_map = [], {}
    for r in ctrl:
        if len(targets) >= N_TARGETS:
            break
        name, seq = r["name"], r["seq"]
        if not (LEN_MIN <= len(seq) <= LEN_MAX):
            continue
        pdb = STRUCT_DIR / f"{name}.pdb"
        if not pdb.exists():
            continue
        try:
            letters, rsa = per_residue_rsa(pdb)
        except Exception as e:
            print(f"  skip {name}: RSA failed ({e})")
            continue
        if len(rsa) != len(seq):
            print(f"  skip {name}: structure {len(rsa)} != seq {len(seq)}")
            continue
        surf = [i for i, v in enumerate(rsa) if not np.isnan(v) and v >= RSA_CUT]
        if len(surf) / len(seq) > MAX_SURF_FRAC:
            print(f"  skip {name}: {len(surf)}/{len(seq)} surface (disordered/extended)")
            continue
        targets.append(dict(name=name, seq=seq, structure_path=str(pdb),
                            n_res=len(seq), n_surface=len(surf)))
        surf_map[name] = {"surface": surf, "len": len(seq)}

    if len(targets) < N_TARGETS:
        print(f"WARNING: only {len(targets)} targets passed QC (wanted {N_TARGETS})")
    pd.DataFrame(targets).to_csv(OUT / "secreted_targets.csv", index=False)
    json.dump(surf_map, open(OUT / "secreted_surface_positions.json", "w"))
    print(f"wrote {len(targets)} targets -> secreted_targets.csv + secreted_surface_positions.json")
    print(pd.DataFrame(targets)[["name", "n_res", "n_surface"]].to_string(index=False))


if __name__ == "__main__":
    main()
