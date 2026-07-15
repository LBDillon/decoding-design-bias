"""
LOCKED evaluator (criteria: docs/EVALUATION_LOCKED.md). Designs held-out TEST backbones with
base vs fine-tuned ProteinMPNN, scores the alkaliphile-secretome axis, applies the pre-registered
PASS/PARTIAL/FAIL rules. Decisive test = NEUTRALOPHILE backbones.

  python alkmpnn/evaluate.py --mpnn third_party/ProteinMPNN \
     --base   third_party/ProteinMPNN/vanilla_model_weights/v_48_002.pt \
     --ft_alk outputs/alkaliphile_v1/model_weights/epoch_00.pt \
     --ft_neu outputs/neutralophile_control_v1/model_weights/epoch_00.pt [--n 8 --temp 0.1 --smoke]

Writes outputs/evaluation/{axis_eval_designs.csv, verdict.txt}.
"""
import sys, argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpnn", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--ft_alk", required=True); ap.add_argument("--ft_neu", default="")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cohort", default="alkaline", choices=["alkaline", "acid"])
    a = ap.parse_args()
    U.set_cohort(a.cohort)
    sys.path.insert(0, a.mpnn)
    device = U.pick_device(); print("device:", device)
    models = {"base": U.load_model(a.base, device), "FT_alk": U.load_model(a.ft_alk, device)}
    if a.ft_neu: models["FT_neu"] = U.load_model(a.ft_neu, device)
    bb = U.collect_backbones("test")
    if a.smoke: bb = bb[:3] + bb[-2:]
    print(f"evaluating {len(bb)} test backbones x {len(models)} models x {a.n} designs")

    rows = []
    for acc, grp, pdb in bb:
        for mname, model in models.items():
            with torch.no_grad(): agg = U.per_backbone(model, pdb, a.n, a.temp, device)
            agg.update(acc=acc, group=grp, model=mname); rows.append(agg)
        print(f"  {acc} ({grp}) done", flush=True)
    df = pd.DataFrame(rows)
    outdir = U.OUTPUTS / "evaluation"; outdir.mkdir(parents=True, exist_ok=True)
    # smoke runs write a SEPARATE file so they can never clobber the locked CSV that backs verdict.txt
    csv = outdir / ("axis_eval_designs_smoke.csv" if a.smoke else "axis_eval_designs.csv")
    df.to_csv(csv, index=False)
    if a.smoke:
        print(f"[smoke] wrote {csv.name} ({df.acc.nunique()} backbones) - NOT the locked CSV/verdict")
        print(df.groupby("model")[U.AXIS + ["recovery"]].mean()); return
    verdict(df, outdir)


def verdict(df, outdir):
    gap = U.natural_gap()
    lines = ["LOCKED-criteria verdict (docs/EVALUATION_LOCKED.md)\n"]
    piv = df[df.group == "neutralophile"].pivot(index="acc", columns="model")
    pvals = {}
    for k in U.AXIS:
        d = (piv[(k, "FT_alk")] - piv[(k, "base")]).dropna()
        try: _, p = wilcoxon(d, alternative="less" if U.DIRECTION[k] < 0 else "greater")
        except Exception: p = np.nan
        frac = (np.sign(d) == U.DIRECTION[k]).mean(); shift = d.mean()
        fgap = (-shift / gap[k]) if gap.get(k) else np.nan   # fraction of gap toward alkaliphile
        pvals[k] = (shift, fgap, frac, p)
    order = sorted(pvals, key=lambda k: pvals[k][3]); mlen = len(order)
    holm = {k: min(1.0, pvals[k][3] * (mlen - i)) for i, k in enumerate(order)}
    lines.append(f"{'metric':<24}{'shift':>9}{'%gap':>8}{'%dir':>7}{'p_raw':>10}{'p_holm':>10}")
    for k in U.AXIS:
        s, fg, fr, p = pvals[k]
        lines.append(f"{k:<24}{s:9.4f}{100*fg:7.0f}%{100*fr:6.0f}%{p:10.2e}{holm[k]:10.2e}")
    rb, rf = piv[("recovery", "base")], piv[("recovery", "FT_alk")]
    nb, nf = piv[("nll", "base")], piv[("nll", "FT_alk")]
    drec = (rf - rb).mean(); dnll = (nf - nb).mean()
    lines.append(f"\nrecovery base {rb.mean():.3f} -> FT_alk {rf.mean():.3f} (Δ {drec:+.3f} pp-frac)")
    lines.append(f"native NLL base {nb.mean():.3f} -> FT_alk {nf.mean():.3f} (Δ {dnll:+.3f} nats)")
    for k in U.OFF_AXIS:
        lines.append(f"off-axis {k}: Δ {(piv[(k,'FT_alk')]-piv[(k,'base')]).dropna().mean():+.4f}")
    _shift = lambda k: (piv[(k, "FT_alk")] - piv[(k, "base")]).dropna().mean()
    if ("core_net", "FT_alk") in piv:   # surface-vs-core specificity (steelman #2)
        lines.append(f"\nspecificity: surface_net Δ {_shift('surface_net'):+.4f} vs core_net Δ {_shift('core_net'):+.4f} "
                     f"| surface_acidic Δ {_shift('surface_acidic'):+.4f} vs core_acidic Δ {_shift('core_acidic'):+.4f} "
                     f"(surface-targeted if |surface| >> |core|)")
    if ("aa_entropy", "FT_alk") in piv:   # collapse audit (R5)
        lines.append(f"collapse: aa-entropy base {piv[('aa_entropy','base')].mean():.2f} -> FT_alk {piv[('aa_entropy','FT_alk')].mean():.2f} bits "
                     f"| diversity base {piv[('diversity','base')].mean():.3f} -> FT_alk {piv[('diversity','FT_alk')].mean():.3f}")
    if "FT_neu" in df.model.unique():
        dn = (piv[("surface_net", "FT_neu")] - piv[("surface_net", "base")]).dropna()
        lines.append(f"\nsymmetric FT_neu surface_net shift {dn.mean():+.4f} "
                     f"(vs FT_alk {pvals['surface_net'][0]:+.4f})")
    s, fg, fr, p = pvals["surface_net"]
    primary = (holm["surface_net"] < 0.05) and (fg > 0)
    magnitude = fg >= 0.25
    guard = (drec >= -0.03) and (dnll <= 0.10)
    decision = "PASS" if (primary and magnitude and guard) else ("PARTIAL" if (primary and guard) else "FAIL")
    lines.append(f"\nPRIMARY(surface_net) p_holm {holm['surface_net']:.2e} dir {'ok' if fg>0 else 'WRONG'} "
                 f"| magnitude {100*fg:.0f}% of gap (>=25%? {magnitude}) | guardrail {'ok' if guard else 'FAIL'}")
    lines.append(f"\n==> DECISION: {decision}")
    txt = "\n".join(lines); print(txt); (outdir / "verdict.txt").write_text(txt)


if __name__ == "__main__":
    main()
