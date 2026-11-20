"""Draw Supplementary Figure S3 from the PDB-control results.

ANALYSIS_DIR is the output of reanalyse_pdb_control.py, or the matching Source Data
folder. The plotted points and display settings are written to OUTPUT_DIR.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--analysis-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(args.analysis_dir / "primary_comparison.csv")
corr = pd.read_csv(args.analysis_dir / "model_vector_correlations.csv")
plt.rcParams.update({"font.family": "Arial", "font.size": 6.3, "axes.labelsize": 6.3, "axes.titlesize": 7,
                     "xtick.labelsize": 6.1, "ytick.labelsize": 6.1, "legend.fontsize": 6.1,
                     "axes.linewidth": .5, "pdf.fonttype": 42})
colours = {"structural input": "#2E7D32", "MIF": "#F9A825", "MIF-ST": "#C0CA33", "sequence mode": "#00838F"}
model_markers = {"ProteinMPNN": "o", "SolubleMPNN": "s", "Caliby": "D", "ESM-IF": "^", "ESM3-struct": "v",
                 "MIF": "s", "MIF-ST": "D", "ESM3-seq": "o", "ESM2-15B": "s", "CARP-640M": "D",
                 "ProGen2-XL": "^", "ProtGPT2": "v"}


def group(row):
    if row.model in ["MIF", "MIF-ST"]:
        return row.model
    return "sequence mode" if row.context == "sequence" else "structural input"


d["display_group"] = d.apply(group, axis=1)
d["display_marker"] = d.model.map(model_markers)
fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 94 / 25.4))
fig.subplots_adjust(left=.085, right=.985, top=.88, bottom=.29, wspace=.30)
panels = [
    ("residual_species_raw_AFmatched", "residual_species_raw_PDB", "Matched AFDB samples",
     "Matched AFDB: species ΔR²", "PDB: species ΔR²", "matched_AF_mean", "residual_species_raw"),
    ("residual_species_adj_AFfull", "residual_species_adj_PDB", "Full AFDB comparison",
     "Full AFDB: adjusted species ΔR²", "PDB: adjusted species ΔR²", "full_AF", "residual_species_adj"),
]
for i, (ax, (x, y, title, xlabel, ylabel, comparison, metric)) in enumerate(zip(axes, panels)):
    high = max(d[x].max(), d[y].max()) * 1.18
    low = min(0, float(min(d[x].min(), d[y].min()))) - .012
    ax.set(xlim=(low, high), ylim=(low, high), xlabel=xlabel, ylabel=ylabel, title=title)
    ax.plot([low, high], [low, high], "k--", lw=.5, alpha=.5, zorder=1)
    ax.grid(color=".92", lw=.35)
    ax.set_axisbelow(True)
    for _, r in d.iterrows():
        ax.scatter(r[x], r[y], s=18, color=colours[r.display_group], marker=r.display_marker,
                   edgecolors=".15", linewidths=.3, zorder=4)
    c = corr.loc[corr.scenario.eq("per_model_matched") & corr.comparison.eq(comparison)
                 & corr.metric.eq(metric)].iloc[0]
    tx, ty, ha, va = (.96, .04, "right", "bottom") if i == 0 else (.04, .96, "left", "top")
    ax.text(tx, ty, f"Spearman ρ={c.spearman_rho:.2f}\np={c.spearman_p_nominal:.3f}; Pearson r={c.pearson_r:.2f}",
            transform=ax.transAxes, ha=ha, va=va, fontsize=6.1, bbox=dict(fc="white", ec=".7", lw=.4, pad=2.0))
    ax.text(-.13, 1.07, "ab"[i], transform=ax.transAxes, weight="bold", fontsize=7)
handles = [Line2D([], [], marker=r.display_marker, color=colours[r.display_group], markeredgecolor=".15",
                  markeredgewidth=.3, linestyle="", markersize=3.8, label=r.model) for _, r in d.iterrows()]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.52, .015),
           columnspacing=2.0, handletextpad=.5)
fig.savefig(args.output_dir / "supplementary_figure_3.pdf")
fig.savefig(args.output_dir / "supplementary_figure_3.png", dpi=250)
plt.close(fig)

d.to_csv(args.output_dir / "Figure_S3_exact_points.csv", index=False)
definition = {
    "layout": "two scatter panels; model colour/shape key below, without labels over points",
    "width_mm": 180, "height_mm": 94, "font_sizes_pt": [6.1, 7.0],
    "left": "rank-corrected raw residual species increments; PDB versus means of 30 controls matched "
            "to each model actual PDB n/domain counts",
    "right": "rank-corrected adjusted R2 increments; PDB versus same-score full AF input",
    "unverified_duplicate_omitted": "SolubleCaliby",
    "correlations": "descriptive across 12 related selected models; p-values nominal",
    "colours": colours, "model_markers": model_markers,
}
(args.output_dir / "Figure_S3_display_definition.json").write_text(json.dumps(definition, indent=2) + "\n")
print(f"Rendered Supplementary Figure S3 in {args.output_dir}")
