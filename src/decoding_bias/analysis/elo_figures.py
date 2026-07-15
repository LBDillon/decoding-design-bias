"""
Regenerate the ELO notebook's publication-style figures for an arbitrary
set of models. Adapted from notebooks/ELO_analysis.ipynb cells 13-14.

Produces in OUTPUT_DIR/figures/:
    - <model>_elo_clean_violin.html : per-model violin of (rating - baseline) by domain
    - elo_heatmap_normalized.html   : interactive heatmap with grouping/normalization dropdowns

Run as a script:
    python -m src.analysis.elo_figures \
        --ratings outputs/elo_analysis_10models/results/all_models_species_ratings_long.csv \
        --dataset data/Decoding_Bias_Dataset.csv \
        --output-dir outputs/elo_analysis_10models
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


# Display labels for every score column we currently score. Extend as
# models get added upstream.
MODEL_LABELS: Dict[str, str] = {
    "proteinmpnn_score":       "ProteinMPNN",
    "esmif_score":             "ESM-IF",
    "mif_score":               "MIF",
    "mifst_score":             "MIF-ST",
    "ESM2_15B_pppl_score":     "ESM2-15B",
    "carp_640M_score":         "CARP-640M",
    "caliby_score":            "Caliby",
    "triflow_score":           "TriFlow",
    "esm3_struct_cond_score":  "ESM3-struct",
    "esm3_seq_only_score":     "ESM3-seq",
    "solublempnn_score":       "SolubleMPNN",
    "soluble_caliby_score":    "SolubleCaliby",
    "progen2_score":           "ProGen2",
    "protgpt2_score":          "ProtGPT2",
    "ProteinMPNN_v020_score":  "ProteinMPNN(v020)",
    "ProteinMPNN_v002_score":  "ProteinMPNN(v002)",
    "AlkSecMPNN_020_score":  "AlkSecMPNN_020",
    "AcidSecMPNN_020_score":"AcidSecMPNN_020",
    "AlkSecMPNN_v2_score":   "AlkSecMPNN(002)",
    "AcidSecMPNN_score":    "AcidSecMPNN(002)",
    # Kept for back-compat with earlier notebook conventions.
    "carp640M_mean_logp":      "CARP-640M",
    "sequence_score_20":       "AlkSecMPNN",
}

# Column order for the heatmap. Groups by architecture family.
MODEL_ORDER: List[str] = [
    "ProteinMPNN", "SolubleMPNN", "ESM-IF", "Caliby", "SolubleCaliby", "TriFlow",
    "ESM3-struct", "MIF", "MIF-ST",
    "ESM2-15B", "CARP-640M", "ESM3-seq", "ProGen2", "ProtGPT2",
    "ProteinMPNN(v020)", "ProteinMPNN(v002)",
    "AlkSecMPNN_020", "AcidSecMPNN_020", "AlkSecMPNN(002)", "AcidSecMPNN(002)",
    "AlkSecMPNN",
]

DOMAIN_COLORS = {
    "Bacteria":  "#2C7BB6",
    "Archaea":   "#D7191C",
    "Eukaryota": "#1A9641",
}


# ---------------------------------------------------------------------------
# Clean violin plots (cell 14)
# ---------------------------------------------------------------------------

def plot_elo_distribution_clean(
    df: pd.DataFrame,
    output_dir: Path,
    baseline_rating: int = 1500,
) -> List[Path]:
    """Plotly violin of (rating - baseline) split by domain, one plot per model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()
        model_df["rating_diff"] = model_df["rating"] - baseline_rating
        model_df = model_df.sort_values(["domain", "rating_diff"])

        fig = go.Figure()
        for domain in sorted(model_df["domain"].unique()):
            sub = model_df[model_df["domain"] == domain]
            fig.add_trace(go.Violin(
                y=sub["rating_diff"],
                name=domain.title(),
                marker_color=DOMAIN_COLORS.get(domain, "#999999"),
                opacity=0.7,
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                points=False,
                bandwidth=30,
            ))

        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2, opacity=0.4)

        y_max = max(abs(model_df["rating_diff"].min()),
                    abs(model_df["rating_diff"].max())) * 1.15

        display_label = MODEL_LABELS.get(model, model)
        fig.update_layout(
            title=f"{display_label} - ELO Rating Differences Distribution by Domain",
            xaxis=dict(title="Domain"),
            yaxis=dict(
                title="Rating Difference",
                range=[-y_max, y_max],
                gridwidth=1, gridcolor="rgba(128,128,128,0.2)",
            ),
            height=600, width=800,
            showlegend=True,
            legend=dict(
                title_text="<b>Domain</b>",
                orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.01,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(128,128,128,0.3)", borderwidth=1,
            ),
            hovermode="closest",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(size=11),
        )

        safe = model.replace("/", "_")
        path = output_dir / f"{safe}_elo_clean_violin.html"
        fig.write_html(path)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Species-level bar chart (interactive port of cell 7's PNG distribution)
# ---------------------------------------------------------------------------

def plot_elo_bar_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    baseline_rating: int = 1500,
) -> List[Path]:
    """One bar per species, colored by domain, sorted by rating.

    Plotly port of the matplotlib bar chart in notebook cell 7. Hover on a
    bar to see the species; dashed lines show each domain's mean rating diff.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_order = ["Archaea", "Bacteria", "Eukaryota"]
    saved: List[Path] = []
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        sub["rating_diff"] = sub["rating"] - baseline_rating
        # group into contiguous domain blocks, sorted within each block
        sub["_dord"] = sub["domain"].map({d: i for i, d in enumerate(domain_order)}).fillna(99)
        sub = sub.sort_values(["_dord", "rating_diff"]).reset_index(drop=True)
        sub["xpos"] = np.arange(len(sub))

        fig = go.Figure()
        for dom in domain_order:
            dsub = sub[sub["domain"] == dom]
            if dsub.empty:
                continue
            fig.add_trace(go.Bar(
                x=dsub["xpos"], y=dsub["rating_diff"], name=dom.title(),
                marker_color=DOMAIN_COLORS.get(dom, "#999999"), opacity=0.8,
                hovertemplate=("Species: %{customdata[0]}<br>Domain: " + dom.title() +
                               "<br>Rating: %{customdata[1]:.0f}<br>"
                               "Diff from 1500: %{y:+.0f}<extra></extra>"),
                customdata=np.stack([dsub["species"], dsub["rating"]], axis=-1),
            ))

        fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.5)

        y_span = max(abs(sub["rating_diff"].min()), abs(sub["rating_diff"].max()))
        y_max = y_span * 1.20
        for k, dom in enumerate(domain_order):
            dsub = sub[sub["domain"] == dom]
            if dsub.empty:
                continue
            x0, x1 = dsub["xpos"].min(), dsub["xpos"].max()
            mean_val = dsub["rating_diff"].mean()
            col = DOMAIN_COLORS.get(dom, "#999999")
            # dashed mean line spanning this domain's block
            fig.add_shape(type="line", x0=x0 - 0.5, x1=x1 + 0.5, y0=mean_val, y1=mean_val,
                          line=dict(color=col, dash="dash", width=1.6), opacity=0.95)
            # mu label box at the bottom of the block
            fig.add_annotation(x=(x0 + x1) / 2, y=-y_max * 0.92,
                               text=f"<b>{dom.title()}</b><br>μ={mean_val:+.0f} (n={len(dsub)})",
                               showarrow=False, align="center",
                               font=dict(size=11, color=col),
                               bordercolor=col, borderwidth=1, borderpad=3,
                               bgcolor="rgba(255,255,255,0.85)")
            # separator after each block except the last
            if k < len(domain_order) - 1:
                fig.add_shape(type="line", x0=x1 + 0.5, x1=x1 + 0.5, y0=-y_max, y1=y_max,
                              line=dict(color="rgba(120,120,120,0.35)", width=1))

        display_label = MODEL_LABELS.get(model, model)
        fig.update_layout(
            title=f"{display_label} - species ELO (− baseline 1500), grouped by domain",
            xaxis=dict(title="Species (grouped by domain, sorted within domain)",
                       showticklabels=False),
            yaxis=dict(title="Difference from Baseline Rating", range=[-y_max, y_max],
                       gridwidth=1, gridcolor="rgba(128,128,128,0.2)", zeroline=False),
            height=600, width=1100, barmode="overlay", showlegend=True,
            legend=dict(title_text="<b>Domain</b>", orientation="v", yanchor="top", y=0.98,
                        xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="rgba(128,128,128,0.3)", borderwidth=1),
            hovermode="closest", plot_bgcolor="white", paper_bgcolor="white",
            font=dict(size=11), margin=dict(r=150),
        )

        safe = model.replace("/", "_")
        path = output_dir / f"{safe}_elo_bar_distribution.html"
        fig.write_html(path)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Interactive heatmap (cell 13)
# ---------------------------------------------------------------------------

def _normalize_matrix(pivot_df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "raw":
        return pivot_df
    if method == "zscore":
        normalized = pivot_df.apply(
            lambda row: stats.zscore(row, nan_policy="omit"),
            axis=1, result_type="broadcast",
        )
        return pd.DataFrame(normalized.values, index=pivot_df.index, columns=pivot_df.columns)
    if method == "minmax":
        def _scale(row):
            if row.max() == row.min():
                return row
            return (row - row.min()) / (row.max() - row.min())
        normalized = pivot_df.apply(_scale, axis=1, result_type="broadcast")
        return pd.DataFrame(normalized.values, index=pivot_df.index, columns=pivot_df.columns)
    return pivot_df


def _make_colorscale(hex_color: str):
    rgb = mcolors.to_rgb(hex_color)
    light = tuple(1 - (1 - c) * 0.15 for c in rgb)
    return [
        [0.0, f"rgb({int(light[0]*255)},{int(light[1]*255)},{int(light[2]*255)})"],
        [1.0, f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})"],
    ]


def _diverging_colorscale(hex_color: str):
    rgb = mcolors.to_rgb(hex_color)
    light = tuple(1 - (1 - c) * 0.3 for c in rgb)
    dark = tuple(c * 0.7 for c in rgb)
    return [
        [0.0,  f"rgb({int(light[0]*255)},{int(light[1]*255)},{int(light[2]*255)})"],
        [0.45, "rgb(250, 250, 250)"],
        [0.5,  "rgb(255, 255, 255)"],
        [0.55, "rgb(250, 250, 250)"],
        [1.0,  f"rgb({int(dark[0]*255)},{int(dark[1]*255)},{int(dark[2]*255)})"],
    ]


def _build_matrix(df: pd.DataFrame, level: str, normalize_method: str):
    if level == "species":
        dfl = (
            df[["species", "model", "rating", "domain", "phylum_use", "class_use"]]
              .rename(columns={"species": "taxon"})
              .copy()
        )
    elif level == "phylum":
        tmp = df[["phylum_use", "model", "rating", "domain"]].rename(columns={"phylum_use": "taxon"})
        dfl = tmp.groupby(["domain", "taxon", "model"], dropna=False)["rating"].mean().reset_index()
    elif level == "class":
        tmp = df[["class_use", "model", "rating", "domain"]].rename(columns={"class_use": "taxon"})
        dfl = tmp.groupby(["domain", "taxon", "model"], dropna=False)["rating"].mean().reset_index()
    else:
        raise ValueError(level)

    dfl["taxon"] = dfl.apply(
        lambda r: f"Other ({r['domain']})" if r["taxon"] == "Other" else r["taxon"], axis=1,
    )
    dfl["model"] = dfl["model"].map(MODEL_LABELS).fillna(dfl["model"])

    domain_order = ["Bacteria", "Archaea", "Eukaryota"]
    dfl["domain_cat"] = pd.Categorical(dfl["domain"], categories=domain_order, ordered=True)
    if level == "species":
        dfl = dfl.sort_values(["domain_cat", "phylum_use", "class_use", "taxon"]).reset_index(drop=True)
    else:
        dfl = dfl.sort_values(["domain_cat", "taxon"]).reset_index(drop=True)

    dfl["row_key"] = dfl["domain"] + " | " + dfl["taxon"]
    pivot = dfl.pivot_table(index="row_key", columns="model", values="rating", aggfunc="mean")
    pivot = pivot.reindex(index=dfl.drop_duplicates(subset=["row_key"])["row_key"].tolist())

    cols = [c for c in MODEL_ORDER if c in pivot.columns] + [c for c in pivot.columns if c not in MODEL_ORDER]
    pivot = pivot.reindex(columns=cols)
    pivot_norm = _normalize_matrix(pivot, method=normalize_method)

    meta = dfl.drop_duplicates(subset=["row_key"]).set_index("row_key")
    dom_vec = meta["domain"].reindex(pivot_norm.index)
    phylum_vec = meta["phylum_use"].reindex(pivot_norm.index) if level == "species" else None
    class_vec = meta["class_use"].reindex(pivot_norm.index) if level == "species" else None

    y_labels = meta["taxon"].reindex(pivot_norm.index).tolist()
    x_labels = pivot_norm.columns.tolist()
    return pivot_norm, dom_vec, y_labels, x_labels, phylum_vec, class_vec


def _grouped_labels(y_labels, group_vec, min_group_size: int = 2):
    groups = []
    current = None
    start = 0
    for idx, g in enumerate(group_vec):
        if g != current:
            if current is not None and (idx - start) >= min_group_size:
                groups.append({"name": current, "start": start, "end": idx - 1,
                               "mid": (start + idx - 1) / 2})
            current = g
            start = idx
    if current is not None and (len(group_vec) - start) >= min_group_size:
        groups.append({"name": current, "start": start, "end": len(group_vec) - 1,
                       "mid": (start + len(group_vec) - 1) / 2})

    tickvals = [g["mid"] for g in groups]
    ticktext = [g["name"] for g in groups]
    shapes = [dict(type="line", x0=-0.5, x1=0,
                   y0=g["end"] + 0.5, y1=g["end"] + 0.5,
                   line=dict(color="white", width=2))
              for g in groups[:-1]]
    return tickvals, ticktext, shapes


def plot_elo_heatmap(
    ratings_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Interactive heatmap with dropdown for Raw/Z-score/Min-Max and grouping by Phylum/Class.

    Expects ratings_df with columns [species, model, rating, domain]
    and taxonomy_df with [species, phylum_division, class].
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tax = taxonomy_df[["species", "phylum_division", "class"]].drop_duplicates("species")
    df = ratings_df.merge(tax, on="species", how="left")
    df["phylum_use"] = df["phylum_division"].fillna("Unknown")
    df["class_use"] = df["class"].fillna("Unknown")

    norm_methods = ["raw", "zscore", "minmax"]
    levels = ["species", "phylum", "class"]

    matrices = {m: {lv: _build_matrix(df, lv, m) for lv in levels} for m in norm_methods}

    _, _, y_species, x_species, species_phylum, species_class = matrices["raw"]["species"]
    _, _, y_phylum, x_phylum, _, _ = matrices["raw"]["phylum"]
    _, _, y_class, x_class, _, _ = matrices["raw"]["class"]

    phylum_tickvals, phylum_ticktext, phylum_shapes = _grouped_labels(
        y_species, species_phylum.fillna("Unknown").tolist(), min_group_size=2)
    class_tickvals, class_ticktext, class_shapes = _grouped_labels(
        y_species, species_class.fillna("Unknown").tolist(), min_group_size=2)

    for shape in phylum_shapes + class_shapes:
        shape["x1"] = len(x_species) - 0.5

    def _height(n):
        return max(600, min(35 * n + 200, 2500))

    heights = {"species": _height(len(y_species)),
               "phylum":  _height(len(y_phylum)),
               "class":   _height(len(y_class))}

    def _scale_range(method):
        if method == "raw":
            dmm = {}
            for dom in ["Bacteria", "Archaea", "Eukaryota"]:
                vals = []
                for lv in levels:
                    pivot = matrices[method][lv][0]
                    dom_vec = matrices[method][lv][1]
                    mask = dom_vec.fillna("Unknown").eq(dom).values
                    mat = pivot.values.copy()
                    mat[~mask, :] = np.nan
                    vals.extend(mat[np.isfinite(mat)])
                dmm[dom] = (min(vals), max(vals)) if vals else (0, 1)
            return dmm
        vals = []
        for lv in levels:
            arr = matrices[method][lv][0].values
            vals.extend(arr[np.isfinite(arr)])
        gr = (min(vals), max(vals)) if vals else (-3, 3)
        return {dom: gr for dom in ["Bacteria", "Archaea", "Eukaryota"]}

    scale_ranges = {m: _scale_range(m) for m in norm_methods}

    colorscales = {
        "raw": {dom: _make_colorscale(color) for dom, color in DOMAIN_COLORS.items()},
        "zscore": {dom: _diverging_colorscale(color) for dom, color in DOMAIN_COLORS.items()},
        "minmax": {dom: _diverging_colorscale(color) for dom, color in DOMAIN_COLORS.items()},
    }

    fig = go.Figure()
    for method in norm_methods:
        for lv in levels:
            pivot, dom_vec, y_labels, x_labels, _, _ = matrices[method][lv]
            for i, dom in enumerate(["Bacteria", "Archaea", "Eukaryota"]):
                mask = dom_vec.fillna("Unknown").eq(dom).values
                mat = pivot.values.copy()
                mat[~mask, :] = np.nan

                visible = (method == "raw" and lv == "species")
                fig.add_trace(go.Heatmap(
                    z=mat, x=x_labels, y=y_labels,
                    colorscale=colorscales[method][dom],
                    zmin=scale_ranges[method][dom][0],
                    zmax=scale_ranges[method][dom][1],
                    zmid=0 if method != "raw" else None,
                    showscale=(i == 0 and lv == "species" and method == "raw"),
                    visible=visible,
                    hovertemplate=(f"{method.upper()}<br>"
                                   "Taxon: %{y}<br>Model: %{x}<br>Value: %{z:.2f}<extra></extra>"),
                    name=f"{lv}_{method}_{dom}",
                ))

    # Build the dropdown
    buttons = []
    for method in norm_methods:
        method_label = {"raw": "Raw ELO", "zscore": "Z-Score", "minmax": "Min-Max"}[method]

        for grouping, tick_info, shapes_list in [
            ("Phylum", (phylum_tickvals, phylum_ticktext), phylum_shapes),
            ("Class", (class_tickvals, class_ticktext), class_shapes),
        ]:
            visible = [False] * len(fig.data)
            start = norm_methods.index(method) * 9
            for i in range(3):
                visible[start + i] = True
            buttons.append(dict(
                label=f"{method_label} - Species ({grouping})",
                method="update",
                args=[{"visible": visible},
                      {"title": f"ELO Heatmap: {method_label} (grouped by {grouping})",
                       "yaxis": {"title": grouping, "tickmode": "array",
                                 "tickvals": tick_info[0], "ticktext": tick_info[1],
                                 "automargin": True, "tickfont": dict(size=10)},
                       "shapes": shapes_list,
                       "height": heights["species"]}],
            ))

        for lv in ["phylum", "class"]:
            visible = [False] * len(fig.data)
            lv_idx = levels.index(lv)
            start = norm_methods.index(method) * 9 + lv_idx * 3
            for i in range(3):
                visible[start + i] = True
            buttons.append(dict(
                label=f"{method_label} - {lv.capitalize()}",
                method="update",
                args=[{"visible": visible},
                      {"title": f"ELO Heatmap: {method_label} by {lv.capitalize()}",
                       "yaxis": {"title": lv.capitalize(), "automargin": True, "tickmode": "linear"},
                       "shapes": [], "height": heights[lv]}],
            ))

    fig.update_layout(
        title="ELO Heatmap: Raw ELO (grouped by Phylum)",
        xaxis=dict(title="Model", tickfont=dict(size=11)),
        yaxis=dict(title="Phylum", tickmode="array",
                   tickvals=phylum_tickvals, ticktext=phylum_ticktext,
                   automargin=True, tickfont=dict(size=10)),
        shapes=phylum_shapes,
        height=heights["species"], width=1400,
        margin=dict(l=220, r=200, t=80, b=60),
        updatemenus=[dict(type="dropdown", x=1.0, xanchor="right", y=1.15, yanchor="top",
                          showactive=True, active=0, buttons=buttons)],
    )
    fig.write_html(output_path)
    return output_path


def plot_elo_heatmap_kingdom(ratings_df: pd.DataFrame, output_path: Path) -> Path:
    """Interactive ELO heatmap grouped by KINGDOM (domain), with the same
    Raw / Z-Score / Min-Max and Species / Kingdom dropdowns as plot_elo_heatmap.

    Use when phylum/class taxonomy is unavailable. Expects ratings_df with
    columns [species, model, rating, domain].
    """
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    df = ratings_df[["species", "model", "rating", "domain"]].copy()
    df["phylum_use"] = df["domain"]; df["class_use"] = df["domain"]  # group by kingdom

    norm_methods = ["raw", "zscore", "minmax"]
    levels = ["species", "phylum"]                  # 'phylum' aggregates to the 3 kingdoms
    lvl_name = {"species": "Species (by kingdom)", "phylum": "Kingdom"}
    matrices = {m: {lv: _build_matrix(df, lv, m) for lv in levels} for m in norm_methods}

    _, _, y_species, x_species, sp_dom, _ = matrices["raw"]["species"]
    k_tickvals, k_ticktext, k_shapes = _grouped_labels(
        y_species, sp_dom.fillna("Unknown").tolist(), min_group_size=2)
    for sh in k_shapes:
        sh["x1"] = len(x_species) - 0.5

    def _h(n):
        return max(500, min(35 * n + 200, 2600))
    heights = {lv: _h(len(matrices["raw"][lv][2])) for lv in levels}

    def _range(method):
        if method == "raw":
            out = {}
            for dom in ["Bacteria", "Archaea", "Eukaryota"]:
                v = []
                for lv in levels:
                    piv, dv = matrices[method][lv][0], matrices[method][lv][1]
                    mat = piv.values.copy()
                    mat[~dv.fillna("Unknown").eq(dom).values, :] = np.nan
                    v.extend(mat[np.isfinite(mat)])
                out[dom] = (min(v), max(v)) if v else (0, 1)
            return out
        v = []
        for lv in levels:
            a = matrices[method][lv][0].values
            v.extend(a[np.isfinite(a)])
        gr = (min(v), max(v)) if v else (-3, 3)
        return {d: gr for d in ["Bacteria", "Archaea", "Eukaryota"]}
    ranges = {m: _range(m) for m in norm_methods}
    cs = {"raw":   {d: _make_colorscale(c) for d, c in DOMAIN_COLORS.items()},
          "zscore": {d: _diverging_colorscale(c) for d, c in DOMAIN_COLORS.items()},
          "minmax": {d: _diverging_colorscale(c) for d, c in DOMAIN_COLORS.items()}}

    fig = go.Figure(); npm = len(levels) * 3
    for method in norm_methods:
        for lv in levels:
            piv, dv, yl, xl, _, _ = matrices[method][lv]
            for i, dom in enumerate(["Bacteria", "Archaea", "Eukaryota"]):
                mat = piv.values.copy()
                mat[~dv.fillna("Unknown").eq(dom).values, :] = np.nan
                fig.add_trace(go.Heatmap(
                    z=mat, x=xl, y=yl, colorscale=cs[method][dom],
                    zmin=ranges[method][dom][0], zmax=ranges[method][dom][1],
                    zmid=0 if method != "raw" else None,
                    showscale=(i == 0 and lv == "species" and method == "raw"),
                    visible=(method == "raw" and lv == "species"),
                    hovertemplate=f"{method.upper()}<br>%{{y}} · %{{x}}<br>%{{z:.2f}}<extra></extra>",
                    name=f"{lv}_{method}_{dom}"))

    buttons = []
    for method in norm_methods:
        ml = {"raw": "Raw ELO", "zscore": "Z-Score", "minmax": "Min-Max"}[method]
        for lv in levels:
            vis = [False] * len(fig.data)
            st = norm_methods.index(method) * npm + levels.index(lv) * 3
            for i in range(3):
                vis[st + i] = True
            yax = ({"title": "Kingdom", "tickmode": "array", "tickvals": k_tickvals,
                    "ticktext": k_ticktext, "automargin": True, "tickfont": dict(size=11)}
                   if lv == "species" else
                   {"title": "Kingdom", "automargin": True, "tickmode": "linear"})
            buttons.append(dict(
                label=f"{ml} - {lvl_name[lv]}", method="update",
                args=[{"visible": vis},
                      {"title": f"Species ELO by kingdom - {ml} ({lvl_name[lv]})",
                       "yaxis": yax, "shapes": (k_shapes if lv == "species" else []),
                       "height": heights[lv]}]))

    fig.update_layout(
        title="Species ELO by kingdom - Raw ELO (Species (by kingdom))",
        xaxis=dict(title="Model", tickfont=dict(size=11), tickangle=-35),
        yaxis=dict(title="Kingdom", tickmode="array", tickvals=k_tickvals,
                   ticktext=k_ticktext, automargin=True, tickfont=dict(size=11)),
        shapes=k_shapes, height=heights["species"], width=1100,
        margin=dict(l=170, r=170, t=90, b=130),
        updatemenus=[dict(type="dropdown", x=1.0, xanchor="right", y=1.12, yanchor="top",
                          showactive=True, active=0, buttons=buttons)])
    fig.write_html(output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratings", required=True, help="all_models_species_ratings_long.csv")
    p.add_argument("--dataset", required=True, help="Decoding_Bias_Dataset.csv (for phylum/class)")
    p.add_argument("--output-dir", required=True, help="Output root (figures/ subfolder written)")
    args = p.parse_args(argv)

    ratings = pd.read_csv(args.ratings).dropna(subset=["species", "model", "rating", "domain"])
    dataset = pd.read_csv(args.dataset).dropna(subset=["species"])

    fig_dir = Path(args.output_dir) / "figures"
    violin_paths = plot_elo_distribution_clean(ratings, fig_dir)
    bar_paths = plot_elo_bar_distribution(ratings, fig_dir)
    heatmap_path = plot_elo_heatmap(ratings, dataset, fig_dir / "elo_heatmap_normalized.html")

    print(f"Violin plots: {len(violin_paths)}")
    print(f"Bar distribution plots: {len(bar_paths)}")
    print(f"Heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
