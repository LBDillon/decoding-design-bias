"""Surface-only ProteinMPNN redesign of the secreted targets, matched to the ESM
run: fix the buried core, redesign only the surface positions (RSA>=0.25), with
base / AlkSec-FT / control-FT weights. Reviewer R1.4 apples-to-apples.

Reuses the committed ProteinMPNN CLI (finetune/third_party/ProteinMPNN). Run with a
torch env, e.g.:
  /Users/lauradillon/miniforge3/envs/esm2_local/bin/python run_proteinmpnn_surface.py

Inputs (from prep_secreted_targets.py): secreted_targets.csv + secreted_surface_positions.json
Output: outputs/esm35m_continual_pretraining/generation/proteinmpnn_designs.csv
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MPNN = ROOT / "finetune" / "third_party" / "ProteinMPNN"
FT = ROOT / "finetune" / "outputs"
GEN = ROOT / "outputs" / "esm35m_continual_pretraining" / "generation"
WORK = GEN / "mpnn_work"
N_DESIGNS = 8
TEMP = "0.1"

# display -> (weights folder, model_name stem)
WEIGHTS = {
    "base":        (MPNN / "vanilla_model_weights", "v_48_020"),
    "AlkSecMPNN":  (FT / "alkaliphile_v1" / "model_weights", "epoch_best"),
    "NeuSecMPNN":  (FT / "neutralophile_control_v1" / "model_weights", "epoch_best"),
}


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def ensure_weights(display, folder, stem):
    """protein_mpnn_run.py expects a 'noise_level' key that the finetune trainer did
    not save; add it (0.2 A training backbone noise) to a patched copy if missing."""
    import torch
    src = Path(folder) / f"{stem}.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "noise_level" in ck:
        return str(folder) + "/", stem
    ck["noise_level"] = 0.2
    wdir = WORK / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    torch.save(ck, wdir / f"{display}.pt")
    return str(wdir) + "/", display


def main():
    targets = pd.read_csv(GEN / "secreted_targets.csv")
    surf = json.load(open(GEN / "secreted_surface_positions.json"))
    WORK.mkdir(parents=True, exist_ok=True)
    pdb_dir = WORK / "pdbs"
    if pdb_dir.exists():
        shutil.rmtree(pdb_dir)
    pdb_dir.mkdir()
    for _, r in targets.iterrows():
        shutil.copy(r["structure_path"], pdb_dir / f"{r['name']}.pdb")

    # 1. parse the PDB backbones
    parsed = WORK / "parsed.jsonl"
    run([sys.executable, MPNN / "helper_scripts" / "parse_multiple_chains.py",
         "--input_path", pdb_dir, "--output_path", parsed], cwd=MPNN)

    # 2. fixed positions = the CORE (1-indexed), so only the surface is redesigned
    fixed = {}
    for _, r in targets.iterrows():
        name = r["name"]
        surf_set = set(surf[name]["surface"])
        core_1idx = [i + 1 for i in range(int(r["n_res"])) if i not in surf_set]
        fixed[name] = {"A": core_1idx}
    fixed_path = WORK / "fixed_positions.jsonl"
    fixed_path.write_text(json.dumps(fixed))

    # 3. design with each weight set (surface-only)
    rows = []
    for model, (folder0, stem0) in WEIGHTS.items():
        folder, stem = ensure_weights(model, folder0, stem0)
        out = WORK / "out" / model
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== ProteinMPNN {model} ({stem}) ===")
        run([sys.executable, MPNN / "protein_mpnn_run.py",
             "--jsonl_path", parsed,
             "--fixed_positions_jsonl", fixed_path,
             "--path_to_model_weights", folder,
             "--model_name", stem,
             "--num_seq_per_target", N_DESIGNS,
             "--sampling_temp", TEMP,
             "--out_folder", out,
             "--batch_size", 1], cwd=MPNN)
        for fa in (out / "seqs").glob("*.fa"):
            name = fa.stem
            recs = _read_fasta(fa)
            # record 0 is the native/input; the rest are the designs
            for k, (_, seq) in enumerate(recs[1:]):
                rows.append(dict(name=name, model=model, sample_idx=k, sequence=seq))

    df = pd.DataFrame(rows)
    df.to_csv(GEN / "proteinmpnn_designs.csv", index=False)
    print(f"\nwrote proteinmpnn_designs.csv "
          f"({len(df)} designs; {df.name.nunique()} targets x {df.model.nunique()} models)")


def _read_fasta(path):
    recs, h, s = [], None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(s)))
            h, s = line[1:], []
        elif line:
            s.append(line)
    if h is not None:
        recs.append((h, "".join(s)))
    return recs


if __name__ == "__main__":
    main()
