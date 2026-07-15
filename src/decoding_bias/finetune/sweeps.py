"""
Model selection / sensitivity sweeps (run BEFORE the locked test eval).

  --mode checkpoint : for each epoch_NN.pt, design VALIDATION neutralophile backbones; find the
                      earliest checkpoint clearing the recovery guardrail (<=3pp) with shift>=25%
                      of the natural gap. (Selection on val, never test.)
  --mode temperature: for one checkpoint, sweep sampling T to see how much of the magnitude is
                      low-T (greedy) amplification vs a real preference.

  python alkmpnn/select.py --mode checkpoint  --mpnn third_party/ProteinMPNN \
     --base third_party/ProteinMPNN/vanilla_model_weights/v_48_002.pt \
     --ckpt_dir outputs/alkaliphile_v1/model_weights --split val
  python alkmpnn/select.py --mode temperature --mpnn third_party/ProteinMPNN \
     --base ... --ckpt outputs/alkaliphile_v1/model_weights/epoch_00.pt
"""
import sys, argparse, glob, os, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U


def eval_model(model, bb, n, temp, device):
    rows = [U.per_backbone(model, pdb, n, temp, device) for _, _, pdb in bb]
    return (np.mean([r["surface_net"] for r in rows]), np.mean([r["recovery"] for r in rows]))


def checkpoint_sweep(a, device, bb, gap):
    base_sn, base_rec = eval_model(U.load_model(a.base, device), bb, a.n, 0.1, device)
    print(f"base: surface_net {base_sn:.4f} recovery {base_rec:.4f}\n")
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "epoch_[0-9]*.pt")),
                   key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    print(f"{'epoch':>6}{'surf_net':>10}{'shift':>9}{'%gap':>7}{'recovery':>10}{'drop_pp':>9}{'guardrail':>10}")
    passing, rows = [], [{"epoch": -1, "surf_net": base_sn, "pct_gap": 0.0, "recovery": base_rec, "drop_pp": 0.0, "pass": "base"}]
    for c in ckpts:
        ep = int(re.search(r"epoch_(\d+)", c).group(1))
        sn, rec = eval_model(U.load_model(c, device), bb, a.n, 0.1, device)
        shift = sn - base_sn; fgap = -shift / gap["surface_net"]; drop = base_rec - rec
        ok = (drop <= a.guardrail) and (fgap >= a.min_gap)
        if ok: passing.append(ep)
        rows.append({"epoch": ep, "surf_net": sn, "pct_gap": 100*fgap, "recovery": rec, "drop_pp": 100*drop, "pass": ok})
        print(f"{ep:>6}{sn:>10.4f}{shift:>9.4f}{100*fgap:>6.0f}%{rec:>10.4f}{100*drop:>8.1f}{('PASS' if ok else '-'):>10}")
    if a.save_csv: pd.DataFrame(rows).to_csv(a.save_csv, index=False); print(f"[saved] {a.save_csv}")
    safe = [r for r in rows if r["epoch"] >= 0 and r["drop_pp"] <= 100 * a.guardrail]
    if a.select == "maxsteer":   # most-trained (= max steer) checkpoint still within the recovery guardrail
        chosen = max(safe, key=lambda r: r["epoch"]) if safe else None
    else:                         # earliest clearing the guardrail with >= min_gap (pre-registered rule)
        elig = [r for r in safe if r["pct_gap"] >= 100 * a.min_gap]
        chosen = min(elig, key=lambda r: r["epoch"]) if elig else None
    if chosen:
        print(f"\nSELECTED (rule={a.select}, guardrail <= {100*a.guardrail:.0f}pp): epoch_{int(chosen['epoch']):02d}"
              f"  ({chosen['pct_gap']:.0f}% of gap, {chosen['drop_pp']:.1f}pp recovery drop)")
        if a.select_out: Path(a.select_out).write_text(f"{int(chosen['epoch']):02d}"); print(f"[selected -> {a.select_out}]")
    else:
        print("\nNo checkpoint within the recovery guardrail; report the tradeoff curve.")


def temperature_sweep(a, device, bb, gap):
    base = U.load_model(a.base, device); ft = U.load_model(a.ckpt, device)
    print(f"{'T':>5}{'base_sn':>10}{'ft_sn':>10}{'shift':>9}{'%gap':>7}{'base_rec':>9}{'ft_rec':>8}")
    rows = []
    for T in [float(t) for t in a.temps.split(",")]:
        bsn, brec = eval_model(base, bb, a.n, T, device)
        fsn, frec = eval_model(ft, bb, a.n, T, device)
        sh = fsn - bsn
        rows.append({"T": T, "pct_gap": -sh/gap['surface_net']*100, "base_rec": brec, "ft_rec": frec, "shift": sh})
        print(f"{T:>5}{bsn:>10.4f}{fsn:>10.4f}{sh:>9.4f}{-sh/gap['surface_net']*100:>6.0f}%{brec:>9.3f}{frec:>8.3f}", flush=True)
    if a.save_csv: pd.DataFrame(rows).to_csv(a.save_csv, index=False); print(f"[saved] {a.save_csv}")


def logitbias_sweep(a, device, bb, gap):
    """Hand-coded logit bias toward acidic (D,E) / away from basic (K,R) on the BASE model - the
    'why not just bias the logits?' baseline. Compare its surface_net-vs-recovery tradeoff to FT_alk."""
    base = U.load_model(a.base, device); ALPH = "ACDEFGHIKLMNPQRSTVWYX"
    def vec(b):
        v = np.zeros(21, np.float32)
        for x in "DE": v[ALPH.index(x)] = b
        for x in "KR": v[ALPH.index(x)] = -b
        return v
    base_sn, rows = None, []
    print(f"{'bias':>6}{'surf_net':>10}{'shift':>9}{'%gap':>7}{'recovery':>10}")
    for b in [float(x) for x in a.biases.split(",")]:
        rs = [U.per_backbone(base, pdb, a.n, 0.1, device, bias_aa=vec(b)) for _, _, pdb in bb]
        sn = float(np.mean([r["surface_net"] for r in rs])); rec = float(np.mean([r["recovery"] for r in rs]))
        if base_sn is None: base_sn = sn
        sh = sn - base_sn
        rows.append({"bias": b, "surf_net": sn, "shift": sh, "pct_gap": -sh/gap["surface_net"]*100, "recovery": rec})
        print(f"{b:>6}{sn:>10.4f}{sh:>9.4f}{-sh/gap['surface_net']*100:>6.0f}%{rec:>10.4f}", flush=True)
    if a.save_csv: pd.DataFrame(rows).to_csv(a.save_csv, index=False); print(f"[saved] {a.save_csv}")
    print("\nRead vs FT_alk (test: surface_net shift ~-0.26 at recovery ~0.53): does the hand-bias need to "
          "sacrifice MORE recovery to reach the same surface_net? If so, the learned readout is better-calibrated.")


def seeds_sweep(a, device, bb, gap):
    """epoch_00 from several seeds, all on the SAME val backbones: is the 1-epoch steer a stable
    property or a lucky single-seed snapshot? Reports surface_net shift + recovery mean +/- SD."""
    base_sn, base_rec = eval_model(U.load_model(a.base, device), bb, a.n, 0.1, device)
    print(f"base: surface_net {base_sn:.4f} recovery {base_rec:.4f}\n")
    ckpts = [c for c in a.ckpts.split(",") if c]
    print(f"{'run':<28}{'surf_net':>10}{'shift':>9}{'%gap':>7}{'recovery':>10}")
    shifts, recs = [], []
    for c in ckpts:
        sn, rec = eval_model(U.load_model(c, device), bb, a.n, 0.1, device)
        sh = sn - base_sn; shifts.append(sh); recs.append(rec)
        run = os.path.basename(os.path.dirname(os.path.dirname(c)))
        print(f"{run:<28}{sn:>10.4f}{sh:>9.4f}{-100*sh/gap['surface_net']:>6.0f}%{rec:>10.4f}")
    shifts, recs = np.array(shifts), np.array(recs); fg = -shifts / gap["surface_net"]
    print(f"\nepoch_0 across {len(ckpts)} seeds: surface_net shift {shifts.mean():.4f} +/- {shifts.std():.4f} "
          f"({100*fg.mean():.0f}% +/- {100*fg.std():.0f}% of gap) | recovery {recs.mean():.4f} +/- {recs.std():.4f} "
          f"(drop {100*(base_rec-recs.mean()):.1f} pp)")
    print("small SD => the published epoch_0 is a stable property, not a single-seed fluke.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["checkpoint", "temperature", "seeds", "logitbias"])
    ap.add_argument("--mpnn", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt_dir"); ap.add_argument("--ckpt")
    ap.add_argument("--ckpts", default="", help="comma-separated epoch_00 checkpoints from different seeds (--mode seeds)")
    ap.add_argument("--save_csv", default="", help="write sweep rows to this CSV (for figures.py process curves)")
    ap.add_argument("--select", default="earliest", choices=["earliest", "maxsteer"],
                    help="earliest = first epoch clearing guardrail w/ >=min_gap (pre-reg); maxsteer = max %gap within guardrail")
    ap.add_argument("--select_out", default="", help="write the selected epoch number (NN) to this file")
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_bb", type=int, default=30)
    ap.add_argument("--guardrail", type=float, default=0.03); ap.add_argument("--min_gap", type=float, default=0.25)
    ap.add_argument("--temps", default="0.1,0.5,1.0")
    ap.add_argument("--biases", default="0,0.5,1,2,3", help="logit-bias strengths (--mode logitbias)")
    ap.add_argument("--cohort", default="alkaline", choices=["alkaline", "acid"])
    a = ap.parse_args()
    U.set_cohort(a.cohort)
    sys.path.insert(0, a.mpnn)
    device = U.pick_device(); gap = U.natural_gap()
    bb = [b for b in U.collect_backbones(a.split) if b[1] == "neutralophile"][:a.max_bb]
    print(f"{a.mode} sweep on {len(bb)} neutralophile {a.split} backbones x n={a.n}; device {device}")
    if a.mode == "checkpoint": checkpoint_sweep(a, device, bb, gap)
    elif a.mode == "seeds": seeds_sweep(a, device, bb, gap)
    elif a.mode == "logitbias": logitbias_sweep(a, device, bb, gap)
    else: temperature_sweep(a, device, bb, gap)


if __name__ == "__main__":
    main()
