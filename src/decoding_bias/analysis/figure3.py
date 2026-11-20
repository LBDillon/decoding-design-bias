"""Figure 3 drawn from the Source Data, without refitting the PCA or GAMs."""
from pathlib import Path
import io
import json
import zipfile
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .gam_display import display_layers  # noqa: E402

MM = 1 / 25.4
DOMAIN_COLOURS = {"Archaea": "#D55E00", "Bacteria": "#0072B2", "Eukaryota": "#009E73"}
# The eight panel-a models, in reading order: (grid-file stem, title).
GAM_PANELS = [("proteinmpnn", "ProteinMPNN"), ("esmif", "ESM-IF"), ("ESM2_15B_pppl", "ESM2-15B"),
              ("mifst", "MIF-ST"), ("caliby", "Caliby"), ("mif", "MIF"), ("protgpt2", "ProtGPT2"),
              ("carp_640M", "CARP-640M")]
LOADING_LABELS = {"isoelectric_point": "Isoelectric point", "basic_residue_fraction": "Basic fraction",
                  "rco": "Relative contact\norder", "instability_index": "Instability index",
                  "acidic_residue_fraction": "Acidic fraction", "avg_cb_distance": "Mean Cβ distance",
                  "sequence_length": "Sequence length", "gravy": "Hydropathicity"}
LOADING_LABEL_POSITIONS = {"isoelectric_point": (-5.45, -.05, "left"),
                           "basic_residue_fraction": (-5.45, 2.6, "left"),
                           "rco": (-5.45, -2.45, "left"), "instability_index": (-1.65, 4.10, "center"),
                           "acidic_residue_fraction": (1.45, 4.25, "center"),
                           "avg_cb_distance": (3.1, 2.6, "left"), "sequence_length": (3.0, 1.05, "left"),
                           "gravy": (2.9, -3.55, "left")}


def reconstruct(source_data, output_dir):
    """Draw Figure 3 from a Source Data ZIP or directory and write verification.json."""
    source_data = Path(source_data)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(source_data) if source_data.is_file() else None
    if archive is None and (source_data / "Source_Data").is_dir():
        source_data = source_data / "Source_Data"

    def read_bytes(name):
        return archive.read("Source_Data/" + name) if archive else (source_data / name).read_bytes()

    def read_csv(name):
        return pd.read_csv(io.BytesIO(read_bytes(name)))

    try:
        plt.rcParams.update({"font.family": "Arial", "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6,
                             "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "legend.fontsize": 5.6,
                             "axes.linewidth": .5, "pdf.fonttype": 42, "ps.fonttype": 42,
                             "svg.fonttype": "none", "savefig.facecolor": "white"})
        folder = "Figure_3bc_and_Table_S14/"
        coords = read_csv(folder + "exact_global_R_coordinates_and_metadata.csv")
        callouts = read_csv(folder + "landmark_coordinates_and_labels.csv")
        loading = read_csv(folder + "top_feature_loading_arrows.csv")
        variance = read_csv(folder + "pca_variance.csv").set_index("PC").explained_variance_pct
        pc1 = f"PC1 ({variance.PC1:.1f}%)"
        pc2 = f"PC2 ({variance.PC2:.1f}%)"
        assert len(coords) == 10146 and coords.Entry.is_unique
        assert len(callouts) == 10 and callouts.Entry.is_unique
        assert len(loading) == 8
        # Panels b-c must use the same coordinate frame as the GAM grids.
        fixed = read_csv("GAM_grids/global_pca_coordinates.csv").set_index("Entry")
        assert np.allclose(coords.set_index("Entry").loc[fixed.index, ["PC1", "PC2"]],
                           fixed[["PC1", "PC2"]], rtol=0, atol=1e-12)
        match = coords.set_index("Entry").loc[callouts.Entry, ["PC1", "PC2"]].to_numpy()
        assert np.allclose(match, callouts[["PC1", "PC2"]], rtol=0, atol=1e-12)

        fig = plt.figure(figsize=(180 * MM, 195 * MM))
        fig.text(.012, .99, "a", weight="bold", fontsize=7, va="top")
        fig.text(.06, .99, "Model-scoring landscapes", weight="bold", fontsize=7, va="top")
        grids = {name: read_csv(f"GAM_grids/{name}_score_grid.csv") for name, _ in GAM_PANELS}
        for k, (name, label) in enumerate(GAM_PANELS):
            table = grids[name]
            assert table.supported.dtype == bool
            x, y, values, opacity, contours = display_layers(table)
            ax = fig.add_axes([.085 + (k % 4) * .225, .748 - (k // 4) * .205, .203, .175])
            mesh = ax.pcolormesh(x, y, values, cmap="viridis", vmin=-3, vmax=1.5, alpha=opacity,
                                 shading="nearest", rasterized=False)
            ax.contour(x, y, contours, colors="white", linewidths=.25, levels=np.arange(-3, 1.6, .5), alpha=.65)
            ax.set(xlim=(-6, 6), ylim=(-6, 6), xticks=[-4, 0, 4], yticks=[-4, 0, 4], title=label)
            ax.set_box_aspect(1)
            if k % 4:
                ax.set_yticklabels([])
            if k < 4:
                ax.set_xticklabels([])
        fig.text(.51, .502, pc1, ha="center", fontsize=6)
        fig.text(.025, .73, pc2, rotation=90, ha="center", fontsize=6)
        cax = fig.add_axes([.30, .477, .55, .011])
        bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        bar.set_ticks([-3, -2, -1, 0, 1])
        fig.text(.285, .482, "Predicted score z-score", ha="right", fontsize=5.5, va="center")

        fig.text(.012, .435, "b", weight="bold", fontsize=7, va="top")
        fig.text(.06, .435, "Protein dataset and landmarks", fontsize=7, weight="bold", va="top")
        ax = fig.add_axes([.09, .075, .39, .31])
        for domain in ["Archaea", "Bacteria", "Eukaryota"]:
            sub = coords[coords.domain.eq(domain)]
            ax.scatter(sub.PC1, sub.PC2, s=.8, color=DOMAIN_COLOURS[domain], alpha=.45, linewidths=0, label=domain)
        for _, r in callouts.iterrows():
            ax.scatter(r.PC1, r.PC2, s=12, c=DOMAIN_COLOURS[r.domain], edgecolors="black", linewidths=.5, zorder=4)
            ax.annotate(r.label + "\n" + r.Entry, (r.PC1, r.PC2), xytext=(r.label_x, r.label_y), fontsize=5.5,
                        ha="center", va="center",
                        bbox={"boxstyle": "round,pad=.18", "fc": "white", "ec": ".6", "lw": .35},
                        arrowprops={"arrowstyle": "-", "lw": .4, "color": ".25"}, zorder=5)
        ax.set(xlim=(-6, 6), ylim=(-7, 8), xlabel=pc1, ylabel=pc2, xticks=[-4, 0, 4], yticks=[-4, 0, 4])
        ax.grid(color=".93", lw=.35)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, markerscale=3,
                  handletextpad=.15, columnspacing=.6)

        fig.text(.52, .435, "c", weight="bold", fontsize=7, va="top")
        fig.text(.57, .435, "Top feature loadings", fontsize=7, weight="bold", va="top")
        ax = fig.add_axes([.59, .075, .375, .31])
        ax.hexbin(coords.PC1, coords.PC2, gridsize=42, mincnt=1, cmap="Greys", alpha=.45, linewidths=.10,
                  edgecolors=".84", zorder=1)
        for _, r in loading.iterrows():
            ax.annotate("", (r.display_x, r.display_y), (0, 0),
                        arrowprops={"arrowstyle": "-|>", "color": ".15", "lw": .8}, zorder=4)
            lx, ly, ha = LOADING_LABEL_POSITIONS[r.feature]
            ax.annotate(LOADING_LABELS[r.feature], (r.display_x, r.display_y), xytext=(lx, ly),
                        textcoords="data", fontsize=5.5, ha=ha, va="center",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .7},
                        arrowprops={"arrowstyle": "-", "lw": .35, "color": ".4", "shrinkA": 2, "shrinkB": 2},
                        zorder=6)
        ax.scatter([0], [0], s=18, color="black", edgecolors="white", linewidths=.35, zorder=8)
        ax.set(xlim=(-6, 6), ylim=(-7, 8), xlabel=pc1, ylabel=pc2, xticks=[-4, 0, 4], yticks=[-4, 0, 4])
        ax.grid(color=".94", lw=.35)
        ax.set_axisbelow(True)
        fig.savefig(out / "Figure_3.pdf")
        plt.close(fig)

        report = {"protein_rows": len(coords), "landmarks": len(callouts), "loading_arrows": len(loading),
                  "fixed_global_coordinate_frame_verified": True, "landmark_coordinates_verified": True,
                  "models": [name for name, _ in GAM_PANELS], "grid_rows": {}, "source_sha256": {},
                  "fits_rerun": False, "numerical_data_changed": False}
        for name, _ in GAM_PANELS:
            relative = f"GAM_grids/{name}_score_grid.csv"
            assert len(grids[name]) == 25600
            report["grid_rows"][name] = len(grids[name])
            report["source_sha256"][relative] = hashlib.sha256(read_bytes(relative)).hexdigest()
        (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report
    finally:
        if archive is not None:
            archive.close()
