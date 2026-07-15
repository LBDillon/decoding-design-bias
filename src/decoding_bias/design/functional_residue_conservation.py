"""
R1.2: do designs PRESERVE catalytic/binding residues, or discard them?

For each design-set protein we pull UniProt functional-site features
(Active site, Binding site, Metal binding, Site, DNA/Nucleotide binding) → the
WT residue positions that matter for function. Designs are full-length redesigns
(no indels), so WT position i maps to design position i. Per design we compute
recovery at FUNCTIONAL positions vs BACKGROUND positions; per model we test
whether functional-site recovery exceeds background (preservation) across proteins.

Reassuring if functional > background (models keep function); concerning if not
(could explain biophysical shifts). Compares structure- vs sequence-conditioned.
"""
import sys, json, time, urllib.request, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = next((p for p in [HERE, *HERE.parents]
                  if (p / "design" / "outputs" / "all_designs_and_wt.csv").exists()),
                 HERE)
OUT = REPO_ROOT / "design" / "outputs"
ALL = OUT / "all_designs_and_wt.csv"
FT_DESIGNS = OUT / "designs_ph_features.csv"
CACHE = OUT / "_uniprot_features_cache.json"
FUNC_TYPES = {"Active site", "Binding site", "Metal binding", "Site",
              "DNA binding", "Nucleotide binding"}
FT_MODELS = ["AlkSecMPNN", "AcidSecMPNN"]
MODEL_ORDER = ["ESM-IF", "MIF", "ProteinMPNN", "AlkSecMPNN", "AcidSecMPNN", "SolubleMPNN",
               "SolubleCaliby", "Caliby", "MIF-ST"]


def p_label(p):
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def plot_summary(summary, out_path):
    summary = summary.copy()
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model")

    x = np.arange(len(summary))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.9, 4.85))
    ax.bar(x - width / 2, summary["func_rec"], width,
           color="#c9282d", label="functional sites")
    ax.bar(x + width / 2, summary["bg_rec"], width,
           color="#9f9f9f", label="background")

    for i, row in enumerate(summary.itertuples(index=False)):
        top = max(row.func_rec, row.bg_rec)
        ax.text(i, top + 0.035,
                f"Δ{row.delta:+.0%}\n{p_label(row.p)}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=35, ha="right")
    ax.set_ylabel("WT-sequence recovery")
    ax.set_ylim(0, max(0.82, float(summary[["func_rec", "bg_rec"]].max().max()) + 0.14))
    ax.set_title(
        "Recovery at annotated functional residues vs background (n=15 templates)\n"
        "No significant positive functional-site preservation"
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fetch_sites(acc, cache):
    if acc in cache:
        return cache[acc]
    try:
        d = json.load(urllib.request.urlopen(
            f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=25))
        seq = d.get("sequence", {}).get("value", "")
        pos = set()
        for f in d.get("features", []):
            if f["type"] in FUNC_TYPES:
                loc = f["location"]
                s = loc["start"].get("value"); e = loc["end"].get("value")
                if s and e:
                    pos.update(range(int(s), int(e) + 1))   # 1-based inclusive
        cache[acc] = {"seq_len": len(seq), "func_pos": sorted(pos)}
    except Exception as ex:
        cache[acc] = {"seq_len": 0, "func_pos": [], "error": str(ex)}
    time.sleep(0.1)
    return cache[acc]


def load_designs():
    c = pd.read_csv(ALL)
    des = c[~c.is_wt].copy()
    needed = ["model", "uniprot_id", "sequence", "wt_sequence"]
    des = des[needed]

    if FT_DESIGNS.exists():
        ft = pd.read_csv(FT_DESIGNS)
        missing = {"model", "uniprot_id", "designed_sequence", "wt_sequence"} - set(ft.columns)
        if missing:
            raise ValueError(f"{FT_DESIGNS} is missing required columns: {sorted(missing)}")
        to_add = [m for m in FT_MODELS if m not in set(des["model"])]
        ft = ft[ft["model"].isin(to_add)].copy()
        if not ft.empty:
            ft["sequence"] = ft["designed_sequence"]
            des = pd.concat([des, ft[needed]], ignore_index=True)

    return des


def main():
    des = load_designs()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # gather functional sites per protein
    meta = {}
    for uid, g in des.groupby("uniprot_id"):
        info = fetch_sites(uid, cache)
        wt = g.iloc[0].wt_sequence
        # only use proteins where UniProt length matches our WT (positions align)
        ok = info["seq_len"] == len(wt) and len(info["func_pos"]) > 0
        meta[uid] = {"func_pos": [p-1 for p in info["func_pos"] if p-1 < len(wt)],
                     "len_match": info["seq_len"] == len(wt),
                     "n_sites": len(info["func_pos"]), "usable": ok}
    CACHE.write_text(json.dumps(cache))

    usable = [u for u, mm in meta.items() if mm["usable"]]
    print(f"Proteins with annotated functional sites + matching length: "
          f"{len(usable)}/{des.uniprot_id.nunique()}")
    print("  (others: no curated catalytic/binding sites, e.g. ribosomal/structural,"
          " or sequence-length mismatch)")

    # per design: recovery at functional vs background positions
    rows = []
    for r in des.itertuples():
        mm = meta.get(r.uniprot_id)
        if not mm or not mm["usable"]:
            continue
        s, wt = r.sequence, r.wt_sequence
        L = min(len(s), len(wt))
        func = [i for i in mm["func_pos"] if i < L]
        if not func:
            continue
        fset = set(func)
        fr = np.mean([s[i] == wt[i] for i in func])
        bg_idx = [i for i in range(L) if i not in fset]
        bg = np.mean([s[i] == wt[i] for i in bg_idx]) if bg_idx else np.nan
        rows.append(dict(model=r.model, uniprot_id=r.uniprot_id,
                         func_recovery=fr, bg_recovery=bg, n_func=len(func)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "functional_residue_recovery.csv", index=False)

    # per (model, protein) mean, then per-model paired test func vs background
    pp = (d.groupby(["model", "uniprot_id"])[["func_recovery", "bg_recovery"]]
          .mean().reset_index().rename(columns={"func_recovery": "func", "bg_recovery": "bg"}))
    print(f"\n{'model':14}{'func-site rec':>14}{'background rec':>15}{'Δ(func−bg)':>12}{'p':>10}{'n_prot':>7}")
    out = []
    for model, g in pp.groupby("model"):
        diff = (g.func - g.bg).values
        p = stats.wilcoxon(diff, alternative="two-sided").pvalue if len(diff) >= 5 else np.nan
        out.append(dict(model=model, func_rec=g.func.mean(), bg_rec=g.bg.mean(),
                        delta=diff.mean(), p=p, n_proteins=len(g)))
        print(f"{model:14}{g.func.mean():13.1%}{g.bg.mean():14.1%}{diff.mean():+11.1%}"
              f"{(f'{p:.1e}' if p==p else 'NA'):>10}{len(g):7}")
    summary = pd.DataFrame(out)
    summary.to_csv(OUT / "functional_residue_conservation_by_model.csv", index=False)
    plot_summary(summary, OUT / "fig_functional_residue_conservation.png")
    print("\nΔ>0 = functional sites preserved ABOVE background (models keep function).")
    print("Wrote functional_residue_recovery.csv + "
          "functional_residue_conservation_by_model.csv + "
          "fig_functional_residue_conservation.png")


if __name__ == "__main__":
    main()
