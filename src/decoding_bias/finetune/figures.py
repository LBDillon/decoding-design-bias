"""
Per-cohort diagnostic figures for the polar fine-tune - built from artifacts already on disk
(no GPU). Reads training logs, the per-backbone eval CSV, the checkpoint sweep and the natural-gap
denominator; writes cohort-prefixed PNGs to finetune/figures/.

  python alkmpnn/figures.py --cohort alkaline    # alkaliphile_fig1..  (from the polar run dir)
  python alkmpnn/figures.py --cohort acid        # acidophile_fig1..
  python alkmpnn/figures.py --cohort alkaline --run_dir <dir> --selected_epoch 02

Defaults read the polar run at ~/Downloads/polar_finetuning/outputs (eval = {label}_axis_eval.csv,
sweep = {label}_sweep.csv, selected epoch = {label}_epoch.txt). Direction/labels/gap signs follow
the cohort via utils.set_cohort (alkaline ⇒ acidic/−, acid ⇒ basic/+).

Cross-cohort figures (dataset distributions, model comparison, polar summary) are a separate
script: analysis/polar_figures.py.
"""
import sys, argparse, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U

FIGDIR = U.FT / "figures"
PRETTY = {"pI": "pI", "net_charge_per_residue": "net charge/res", "surface_net": "surface net\n(K+R−D−E)",
          "surface_lys": "surface Lys", "KtoKR": "K/(K+R)"}
LABEL = "alkaliphile"; PREFIX = "alkaliphile"; SEL_EP = 0   # set per-cohort in main()


def _dirword():
    return "acidic (more negative)" if U.DIRECTION["surface_net"] < 0 else "basic (more positive)"


def parse_log(path):
    """log.txt -> DataFrame(epoch, train_nll, train_rec, val_nll, val_rec)."""
    rows = []
    for ln in Path(path).read_text().splitlines():
        t = ln.replace("|", "").split()
        if not t or t[0] != "epoch":
            continue
        d = {t[i]: t[i + 1] for i in range(0, len(t) - 1, 2)}
        rows.append({"epoch": int(d["epoch"]), "train_nll": float(d["train_nll"]),
                     "train_rec": float(d["train_rec"]), "val_nll": float(d["val_nll"]),
                     "val_rec": float(d["val_rec"])})
    return pd.DataFrame(rows)


def fig_training_curves(run_dir):
    runs = {"FT_case": run_dir / f"{LABEL}_v1" / "log.txt",
            "FT_neu": run_dir / f"{LABEL}_neu_v1" / "log.txt"}
    logs = {k: parse_log(p) for k, p in runs.items() if p.exists()}
    if not logs:
        print("  (no log.txt found - skipping training curves)"); return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, df in logs.items():
        ax[0].plot(df.epoch, df.val_nll, marker=".", label=f"{name} val")
        ax[1].plot(df.epoch, df.val_rec, marker=".", label=f"{name} val")
    for a in ax:
        a.axvline(SEL_EP, color="crimson", ls="--", lw=1)
        a.text(SEL_EP + 0.3, a.get_ylim()[0], f"epoch_{SEL_EP:02d}\n(selected)", color="crimson", fontsize=8, va="bottom")
        a.set_xlabel("epoch"); a.legend(fontsize=8)
    ax[0].set_ylabel("validation NLL (nats)")
    ax[0].set_title(f"{LABEL}: val NLL keeps dropping → NLL early-stop\nwould pick the over-trained model")
    ax[1].set_ylabel("validation recovery")
    ax[1].set_title("(training-time metrics: backbone noise on,\nso lower than clean-eval recovery)")
    fig.tight_layout(); _save(fig, "fig1_training_curves.png")


def _load_eval(eval_csv):
    df = pd.read_csv(eval_csv)
    n_neu = df[df.group == "neutralophile"].acc.nunique()
    if n_neu < 10:
        print(f"  warning: eval CSV has only {n_neu} neutralophile backbones - looks like a --smoke run.")
    return df, n_neu


def fig_steering_bars(df):
    gap = U.natural_gap()
    piv = df[df.group == "neutralophile"].pivot(index="acc", columns="model")
    models = [m for m in ("FT_alk", "FT_neu") if m in df.model.unique()]
    pct = {m: [(-(piv[(k, m)] - piv[(k, "base")]).mean() / gap[k]) * 100 for k in U.AXIS] for m in models}
    x = np.arange(len(U.AXIS)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = {"FT_alk": "#c0392b", "FT_neu": "#2980b9"}
    names = {"FT_alk": "FT_case", "FT_neu": "FT_neu (control)"}
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, pct[m], w, label=names[m], color=colors.get(m))
    ax.axhline(100, color="grey", ls="--", lw=1); ax.text(len(x) - 0.5, 102, "100% = natural gap", fontsize=8, ha="right")
    ax.axhline(25, color="green", ls=":", lw=1); ax.text(len(x) - 0.5, 27, "25% pre-reg threshold", fontsize=8, ha="right", color="green")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[k] for k in U.AXIS], fontsize=9)
    ax.set_ylabel(f"design shift toward the {LABEL} axis\n(% of natural gap)")
    ax.set_title(f"Fine-tuning steers unseen neutralophile backbones toward the {LABEL} axis\n"
                 f"(FT_neu = symmetric control)")
    ax.legend(); fig.tight_layout(); _save(fig, "fig2_steering_bars.png")


def fig_paired_surface_net(df):
    piv = df[df.group == "neutralophile"].pivot(index="acc", columns="model")["surface_net"]
    if "FT_alk" not in piv:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for acc, r in piv.iterrows():
        ax.plot([0, 1], [r["base"], r["FT_alk"]], color="#c0392b", alpha=0.4, lw=0.8, marker="o", ms=3)
    ax.plot([0, 1], [piv["base"].mean(), piv["FT_alk"].mean()], color="black", lw=2.5, marker="o", label="mean")
    d = U.DIRECTION["surface_net"]
    frac = (((piv["FT_alk"] - piv["base"]) * d) > 0).mean() * 100
    ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "FT_case"])
    ax.set_ylabel("surface net charge  (K+R−D−E)/n_surf")
    ax.set_title(f"{LABEL}: per-backbone surface charge base → FT_case\n"
                 f"{frac:.0f}% of backbones move {_dirword()}")
    ax.legend(); fig.tight_layout(); _save(fig, "fig3_paired_surface_net.png")


def fig_recovery_guardrail(df):
    piv = df[df.group == "neutralophile"].pivot(index="acc", columns="model")["recovery"]
    if "FT_alk" not in piv:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for acc, r in piv.iterrows():
        ax.plot([0, 1], [r["base"], r["FT_alk"]], color="grey", alpha=0.4, lw=0.8, marker="o", ms=3)
    mb, mf = piv["base"].mean(), piv["FT_alk"].mean()
    ax.plot([0, 1], [mb, mf], color="black", lw=2.5, marker="o", label=f"mean {mb:.3f}→{mf:.3f} ({(mf-mb)*100:+.1f} pp)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "FT_case"])
    ax.set_ylabel("native sequence recovery (argmax)")
    ax.set_title(f"{LABEL}: recovery guardrail (budget −3 pp)\nΔ = {(mf-mb)*100:+.1f} pp → {'PASS' if (mf-mb)>=-0.03 else 'FAIL'}")
    ax.legend(fontsize=8); fig.tight_layout(); _save(fig, "fig4_recovery_guardrail.png")


def fig_surface_vs_core(df):
    """Is the charge change surface-targeted (context-specific) or global? Needs the EXTRA columns."""
    piv = df[df.group == "neutralophile"].pivot(index="acc", columns="model")
    if ("core_net", "FT_alk") not in piv:
        return
    metrics = [("surface_net", "surface\nnet"), ("core_net", "core\nnet"),
               ("surface_acidic", "surface\nacidic"), ("core_acidic", "core\nacidic")]
    sh = [(piv[(k, "FT_alk")] - piv[(k, "base")]).dropna().mean() for k, _ in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(x, sh, color=["#c0392b", "#e8a5a0", "#2980b9", "#a3c6e0"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_ylabel("FT_case − base shift")
    ax.set_title(f"{LABEL}: surface-targeted vs global\nchange concentrated at the surface = context-specific")
    fig.tight_layout(); _save(fig, "fig7_surface_vs_core.png")


def fig_steering_curve(csv):
    """The dial: steering (% of natural gap) and recovery vs training epoch (sweeps --mode checkpoint)."""
    df = pd.read_csv(csv); d = df[df.epoch >= 0]
    fig, ax = plt.subplots(figsize=(8, 4.6)); ax2 = ax.twinx()
    ax.plot(d.epoch, d.pct_gap, "o-", color="#c0392b", label="steering")
    ax2.plot(d.epoch, d.recovery, "s--", color="#2ca02c", alpha=0.8, label="recovery")
    ax.axhline(100, color="grey", ls="--", lw=1); ax.text(d.epoch.max(), 104, "100% = natural gap", fontsize=8, ha="right")
    ax.axvline(SEL_EP, color="crimson", ls=":", lw=1); ax.text(SEL_EP, ax.get_ylim()[1]*0.92, f" epoch_{SEL_EP:02d}", color="crimson", fontsize=8)
    base = df[df.epoch < 0]
    if len(base): ax2.axhline(base.recovery.iloc[0], color="green", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("steering (% of natural gap)", color="#c0392b")
    ax2.set_ylabel("native recovery", color="#2ca02c")
    ax.set_title(f"{LABEL}: steering vs training - the controllable dial\n(steer climbs while recovery holds; selected epoch marked)")
    fig.tight_layout(); _save(fig, "fig5_steering_curve.png")


def fig_temperature_curve(csv):
    """Magnitude vs sampling temperature (sweeps --mode temperature): T=0.1 greedy-amplifies, T=1.0 ≈ honest."""
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 4.6)); ax2 = ax.twinx()
    ax.plot(df["T"], df.pct_gap, "o-", color="#c0392b", label="steering")
    ax2.plot(df["T"], df.ft_rec, "s--", color="#2ca02c", alpha=0.8, label="FT recovery")
    ax.axhline(100, color="grey", ls="--", lw=1); ax.text(df["T"].max(), 104, "100% = natural gap", fontsize=8, ha="right")
    ax.set_xlabel("sampling temperature"); ax.set_ylabel("steering (% of natural gap)", color="#c0392b")
    ax2.set_ylabel("FT recovery", color="#2ca02c")
    ax.set_title(f"{LABEL}: magnitude vs sampling temperature\n(T=0.1 greedy-amplifies; T=1.0 ≈ honest size)")
    fig.tight_layout(); _save(fig, "fig6_temperature_curve.png")


def _save(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    out = f"{PREFIX}_{name}"
    fig.savefig(FIGDIR / out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote figures/{out}")


def main():
    global LABEL, PREFIX, SEL_EP
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="alkaline", choices=["alkaline", "acid"])
    ap.add_argument("--run_dir", default=os.path.expanduser("~/Downloads/polar_finetuning/outputs"),
                    help="root with {label}_v1/ logs + evaluation/{label}_*.csv")
    ap.add_argument("--eval_csv", default="", help="override eval CSV (default run_dir/evaluation/{label}_axis_eval.csv)")
    ap.add_argument("--checkpoint_csv", default="", help="override sweep CSV (default run_dir/evaluation/{label}_sweep.csv)")
    ap.add_argument("--temperature_csv", default="", help="temperature sweep CSV → fig6 (optional)")
    ap.add_argument("--selected_epoch", default="", help="override; default read from {label}_epoch.txt")
    a = ap.parse_args()
    U.set_cohort(a.cohort)
    LABEL = U._cf("label"); PREFIX = LABEL
    run_dir = Path(a.run_dir); evald = run_dir / "evaluation"
    eval_csv = Path(a.eval_csv) if a.eval_csv else evald / f"{LABEL}_axis_eval.csv"
    ckpt_csv = Path(a.checkpoint_csv) if a.checkpoint_csv else evald / f"{LABEL}_sweep.csv"
    ep_file = evald / f"{LABEL}_epoch.txt"
    if a.selected_epoch: SEL_EP = int(a.selected_epoch)
    elif ep_file.exists(): SEL_EP = int(ep_file.read_text().strip())

    print(f"[{LABEL}] figures → finetune/figures/  (selected epoch_{SEL_EP:02d}; run_dir {run_dir})")
    fig_training_curves(run_dir)
    if eval_csv.exists():
        df, _ = _load_eval(eval_csv)
        fig_steering_bars(df); fig_paired_surface_net(df); fig_recovery_guardrail(df); fig_surface_vs_core(df)
    else:
        print(f"  (no eval CSV at {eval_csv})")
    if ckpt_csv.exists(): fig_steering_curve(ckpt_csv)
    else: print(f"  (no checkpoint sweep CSV at {ckpt_csv} - skip fig5)")
    if a.temperature_csv and Path(a.temperature_csv).exists(): fig_temperature_curve(a.temperature_csv)
    else: print("  (no --temperature_csv given - skip fig6; needs a sweeps --mode temperature run)")


if __name__ == "__main__":
    main()
