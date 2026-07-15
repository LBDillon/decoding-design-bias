"""
elo_paper_figures.py

Publication panels for the Elo section, built from the `long` species-ratings
frame (columns: species, model, rating, domain) that run_elo_variants already
writes, plus the taxonomy CSV (species -> phylum_division).

Panels (see the recommended allocation):
  Main Fig 2A  species_bars(...)          per-species bars, 1 structure vs 1 sequence model
  Main Fig 2B  gap_dotplot(...)           Archaea-Eukaryota gap, focused models, bootstrap CI
  Supp  S1A    domain_violin(...)         per-domain violins across 14 models (+ optional points)
  Supp  S1B    phylum_model_heatmap(...)  phylum x model, ordered, single diverging scale

Every panel is written as HTML and as a vector PDF (kaleido). Import and call
build_all(long, tax, figdir, title=...) from the run script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DOM = ["Archaea", "Bacteria", "Eukaryota"]
DOM_LINE = {"Archaea": "#D7191C", "Bacteria": "#2C7BB6", "Eukaryota": "#1A9641"}
DOM_FILL = {"Archaea": "rgba(215,25,28,0.40)", "Bacteria": "rgba(44,123,182,0.40)",
            "Eukaryota": "rgba(26,150,65,0.40)"}

# left-to-right / grouping order by architecture class (matches the paper's 3 regimes)
CLASS_ORDER = [
    "proteinmpnn_score", "solublempnn_score", "caliby_score", "soluble_caliby_score",
    "esmif_score", "triflow_score",                                    # backbone-conditioned (6)
    "mif_score", "mifst_score", "esm3_struct_cond_score",              # structure + visible seq (3)
    "esm3_seq_only_score", "ESM2_15B_pppl_score", "carp_640M_score",
    "progen2_score", "protgpt2_score",                                 # sequence-only (5)
]
_CLS = ({m: "Backbone-conditioned" for m in CLASS_ORDER[:6]}
        | {m: "Structure + visible seq" for m in CLASS_ORDER[6:9]}
        | {m: "Sequence-only" for m in CLASS_ORDER[9:]})
CLS_COLOR = {"Backbone-conditioned": "#2C7BB6",
             "Structure + visible seq": "#7B3294",
             "Sequence-only": "#1A9641"}

PRETTY = {
    "proteinmpnn_score": "ProteinMPNN", "solublempnn_score": "SolubleMPNN",
    "caliby_score": "Caliby", "soluble_caliby_score": "SolubleCaliby",
    "esmif_score": "ESM-IF", "triflow_score": "TriFlow",
    "mif_score": "MIF", "mifst_score": "MIF-ST", "esm3_struct_cond_score": "ESM3-struct",
    "esm3_seq_only_score": "ESM3-seq", "ESM2_15B_pppl_score": "ESM2-15B",
    "carp_640M_score": "CARP-640M", "progen2_score": "ProGen2-XL", "protgpt2_score": "ProtGPT2",
}

# Focused paper-panel selections. ``None`` means all available models in CLASS_ORDER.
FIG2B_MODELS = None
REP_MODELS = ["proteinmpnn_score", "ESM2_15B_pppl_score", "esmif_score",
              "carp_640M_score", "mif_score", "mifst_score"]

# GTDB/NCBI nomenclature collisions in the taxonomy CSV: same phylum, two names.
# Map the NCBI name onto the GTDB name so the heatmap gets one row per phylum.
PHYLUM_ALIASES = {
    "Firmicutes": "Bacillota",
    "Proteobacteria": "Pseudomonadota",
    "Cyanobacteria": "Cyanobacteriota",
    "Actinobacteria": "Actinomycetota",
    "Bacteroidetes": "Bacteroidota",
}


def _style(fig):
    size = fig.layout.font.size or 12
    color = fig.layout.font.color or "#222"
    fig.update_layout(template="simple_white",
                      font=dict(family="Arial, Helvetica, sans-serif", size=size, color=color))
    # only supply a default margin if the panel hasn't set its own
    if fig.layout.margin.l is None:
        fig.update_layout(margin=dict(l=70, r=30, t=60, b=70))
    return fig


def _save(fig, html_path):
    """Write HTML and a same-name vector PDF. Self-contained (no monkey-patch needed)."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    pdf_path = html_path.with_suffix(".pdf")
    try:
        fig.write_image(str(pdf_path), format="pdf")   # vector; scale irrelevant for PDF
    except Exception as ex:
        print(f"  PDF export failed for {pdf_path.name}: {ex}\n"
              "  plotly>=6 ships kaleido>=1, which needs a Chrome:\n"
              "    run  `plotly_get_chrome`   (or  python -c \"import plotly.io as pio; pio.get_chrome()\")\n"
              "  or pin the old exporter:  pip install 'plotly<6' 'kaleido==0.2.1'")
    return html_path


def _models_present(long):
    present = set(long["model"].unique())
    return [m for m in CLASS_ORDER if m in present]


def arch_euk_gap(long):
    means = long.groupby(["model", "domain"])["rating"].mean().unstack()
    return means["Archaea"] - means["Eukaryota"]


def _ordered_models(long, models=None, order_by_delta=False):
    if models is None:
        order = _models_present(long)
    else:
        present = set(long["model"].unique())
        order = [m for m in dict.fromkeys(models) if m in present]
    if order_by_delta:
        gap = arch_euk_gap(long[long.model.isin(order)])
        order = sorted(order, key=lambda m: gap.get(m, -np.inf), reverse=True)
    return order


# ---------- Main Fig 2A: per-species bars, one structure vs one sequence model ----------
def species_bars(long, out, struct="proteinmpnn_score", seq="protgpt2_score", title="",
                 label_extremes=True):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=(PRETTY.get(struct, struct), PRETTY.get(seq, seq)))
    dord = {k: i for i, k in enumerate(DOM)}
    for r, m in enumerate([struct, seq], start=1):
        sub = long[long.model == m].copy()
        sub["_d"] = sub.domain.map(dord)
        sub = sub.sort_values(["_d", "rating"]).reset_index(drop=True)
        sub["x"] = np.arange(len(sub))
        start = 0
        for dom in DOM:
            s = sub[sub.domain == dom]
            if s.empty:
                continue
            fig.add_trace(go.Bar(x=s.x, y=s.rating - 1500, marker_color=DOM_LINE[dom],
                                 marker_line_width=0, name=dom, legendgroup=dom,
                                 showlegend=(r == 1)), row=r, col=1)
            mu = s.rating.mean() - 1500
            fig.add_hline(y=mu, line_dash="dash", line_color=DOM_LINE[dom], line_width=1,
                          row=r, col=1)
            # domain-mean value label, anchored at the block's left edge above/below the line
            fig.add_annotation(x=start + 0.5, y=mu, row=r, col=1, xanchor="left",
                               yanchor="bottom" if mu >= 0 else "top", showarrow=False,
                               text=f"<b>{dom} μ={mu:+.0f}</b>",
                               font=dict(size=9, color=DOM_LINE[dom]),
                               bgcolor="rgba(255,255,255,0.7)")
            start += len(s)
        # vertical separators between domain blocks
        b = 0
        for dom in DOM:
            n = int((sub.domain == dom).sum())
            b += n
            if b < len(sub):
                fig.add_vline(x=b - 0.5, line_color="grey", line_width=0.6, row=r, col=1)
        # label the single highest / lowest species for this model
        if label_extremes:
            for idx, ay in [(sub.rating.idxmax(), -26), (sub.rating.idxmin(), 26)]:
                row_ = sub.loc[idx]
                fig.add_annotation(x=row_.x, y=row_.rating - 1500, row=r, col=1,
                                   text=f"{row_.species} ({row_.rating - 1500:+.0f})",
                                   font=dict(size=8, color="#333"), showarrow=True,
                                   arrowhead=2, arrowsize=0.6, arrowwidth=0.8,
                                   arrowcolor="#666", ax=0, ay=ay)
    fig.update_layout(barmode="overlay", bargap=0, height=680, width=800,
                      title=f"{title}: species Elo by domain (baseline 1500)",
                      legend_title_text="Domain")
    fig.update_yaxes(title_text="Elo − 1500")
    fig.update_xaxes(showticklabels=False, title_text="species (grouped by domain, sorted within)")
    return _save(_style(fig), out)


# ---------- Main Fig 2B: Archaea-Eukaryota gap dot-plot, bootstrap CI ----------
def gap_dotplot(long, out, title="", n_boot=2000, seed=0, models=None,
                order_by_delta=False):
    rng = np.random.default_rng(seed)
    rows = []
    for m in _ordered_models(long, models=models, order_by_delta=order_by_delta):
        sub = long[long.model == m]
        a = sub.loc[sub.domain == "Archaea", "rating"].to_numpy()
        e = sub.loc[sub.domain == "Eukaryota", "rating"].to_numpy()
        gap = a.mean() - e.mean()
        boot = np.array([rng.choice(a, a.size, True).mean() - rng.choice(e, e.size, True).mean()
                         for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append((PRETTY.get(m, m), gap, gap - lo, hi - gap, _CLS.get(m, "")))
    df = pd.DataFrame(rows, columns=["label", "gap", "lo", "hi", "cls"])
    y_order = df["label"].tolist()
    fig = go.Figure()
    for cls in ["Backbone-conditioned", "Structure + visible seq", "Sequence-only"]:
        d = df[df.cls == cls]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d.gap, y=d.label, mode="markers", name=cls,
            marker=dict(size=9, color=CLS_COLOR[cls]),
            error_x=dict(type="data", symmetric=False, array=d.hi, arrayminus=d.lo,
                         thickness=1.2, width=4, color=CLS_COLOR[cls])))
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    # gap ± 95% CI printed as a neat right-hand column (absolute interval)
    for _, r in df.iterrows():
        fig.add_annotation(x=1.015, xref="paper", y=r.label, yref="y", xanchor="left",
                           showarrow=False, font=dict(size=10, color="#333"),
                           text=f"{r.gap:+.0f}  [{r.gap - r.lo:+.0f}, {r.gap + r.hi:+.0f}]")
    fig.add_annotation(x=1.015, xref="paper", y=1.0, yref="paper", yanchor="bottom",
                       xanchor="left", showarrow=False, font=dict(size=10, color="#333"),
                       text="<b>gap [95% CI]</b>")
    fig.update_layout(height=26 * len(df) + 240, width=760,
                      title=f"{title}: Archaea − Eukaryota Elo gap (95% species bootstrap)",
                      xaxis_title="Archaea − Eukaryota Elo gap",
                      yaxis=dict(categoryorder="array", categoryarray=y_order,
                                 autorange="reversed"),
                      margin=dict(l=110, r=170, t=70, b=70),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return _save(_style(fig), out)


# ---------- Supp S1A: per-domain violins across all models ----------
def domain_violin(long, out, title="", points=False, models=None,
                  order_by_delta=False):
    """Per-domain violins. Pass ``models`` (list of score columns) to show only a
    representative subset - the full 14-model version is unreadably dense in one row."""
    order = _ordered_models(long, models=models, order_by_delta=order_by_delta)
    if models is not None:
        long = long[long.model.isin(order)]
    xcats = [PRETTY.get(m, m) for m in order]
    fig = go.Figure()
    for dom in DOM:
        sub = long[long.domain == dom]
        fig.add_trace(go.Violin(
            x=[PRETTY.get(m, m) for m in sub.model], y=sub.rating - 1500, name=dom,
            line_color=DOM_LINE[dom], fillcolor=DOM_FILL[dom],
            box_visible=True, meanline_visible=True, spanmode="hard",
            points=points, jitter=0.30, pointpos=0,
            marker=dict(size=2, opacity=0.30, color=DOM_LINE[dom]),
            scalegroup=dom, legendgroup=dom))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    gap = arch_euk_gap(long)
    ytop = long.rating.max() - 1500
    for m in order:
        fig.add_annotation(x=PRETTY.get(m, m), y=ytop, yshift=10, showarrow=False,
                           text=f"Δ{gap[m]:+.0f}", font=dict(size=9, color="#444"))
    fig.update_layout(violinmode="group", height=640, width=130 * len(order) + 260,
                      title=f"{title}: species Elo (− 1500) by domain",
                      yaxis_title="Elo − 1500", xaxis=dict(tickangle=-40),
                      xaxis_categoryorder="array", xaxis_categoryarray=xcats,
                      legend_title_text="Domain")
    return _save(_style(fig), out)


# ---------- Supp S1B: phylum x model heatmap, static, ordered ----------
def phylum_model_heatmap(long, tax, out, title="", value="z",
                         domain_order=("Eukaryota", "Archaea", "Bacteria")):
    tax = tax[["species", "phylum_division"]].drop_duplicates("species")
    d = long.merge(tax, on="species", how="left")
    d["phylum"] = d["phylum_division"].fillna("Other").replace(PHYLUM_ALIASES)
    if value == "z":
        d["val"] = d.groupby("model")["rating"].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
        cbar, zmid = "Elo z (per model)", 0.0
    else:
        d["val"], cbar, zmid = d["rating"] - 1500, "Elo − 1500", 0.0

    cols = [m for m in CLASS_ORDER if m in d.model.unique()]
    piv = (d.groupby(["phylum", "model"])["val"].mean()
           .reset_index().pivot(index="phylum", columns="model", values="val")[cols])
    phy_dom = d.groupby("phylum")["domain"].agg(lambda s: s.mode().iat[0])

    order = []
    for dm in domain_order:
        block = [p for p in piv.index if phy_dom.get(p) == dm]
        block.sort(key=lambda p: piv.loc[p].mean(), reverse=True)   # within domain, high->low
        order += block
    piv = piv.reindex(order)

    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[PRETTY.get(c, c) for c in piv.columns],
        y=list(range(len(order))), colorscale="RdBu_r", zmid=zmid,
        xgap=1, ygap=1, colorbar=dict(title=cbar),
        hovertext=[[f"{p} · {phy_dom.get(p)}" for _ in piv.columns] for p in order],
        hoverinfo="text+z"))
    # Domain separators. Phylum labels are custom annotations so each row label can
    # inherit its domain color in both the HTML and static PDF exports.
    b = 0
    for dm in domain_order:
        n = sum(phy_dom.get(p) == dm for p in order)
        if n == 0:
            continue
        b += n
        if b < len(order):
            fig.add_hline(y=b - 0.5, line_color="black", line_width=1)

    xlabels = [PRETTY.get(c, c) for c in piv.columns]
    for y, phylum in enumerate(order):
        fig.add_annotation(x=0, xref="paper", y=y, yref="y", xshift=-10,
                           text=phylum, showarrow=False, xanchor="right",
                           yanchor="middle",
                           font=dict(size=13, color=DOM_LINE.get(phy_dom.get(phylum), "#333")))
    for lab in xlabels:
        fig.add_annotation(x=lab, xref="x", y=-0.03, yref="paper",
                           text=lab, showarrow=False, textangle=-70,
                           xanchor="right", yanchor="top",
                           font=dict(size=13, color="#222"))

    cell_px = 24
    margin = dict(l=220, r=130, t=80, b=175)
    fig.update_layout(height=cell_px * len(order) + margin["t"] + margin["b"],
                      width=cell_px * len(piv.columns) + margin["l"] + margin["r"],
                      title=dict(text=f"{title}: phylum × model Elo ({cbar})",
                                 font=dict(size=18)),
                      font=dict(size=14),
                      margin=margin,
                      xaxis=dict(showticklabels=False, title=""),
                      yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                                 showticklabels=False, autorange="reversed",
                                 automargin=False,
                                 title_text=""))
    fig.update_traces(colorbar=dict(title=dict(text=cbar, font=dict(size=14)),
                                    tickfont=dict(size=13)))
    return _save(_style(fig), out)


def top_bottom_species(long, n=3):
    """Tidy table of the n highest- and n lowest-rated species per model (rank 1 = extreme)."""
    recs = []
    for m in _models_present(long):
        sub = long[long.model == m].sort_values("rating")
        for rank, (_, row) in enumerate(sub.tail(n)[::-1].iterrows(), 1):
            recs.append((PRETTY.get(m, m), "top", rank, row.species, row.domain, round(row.rating, 1)))
        for rank, (_, row) in enumerate(sub.head(n).iterrows(), 1):
            recs.append((PRETTY.get(m, m), "bottom", rank, row.species, row.domain, round(row.rating, 1)))
    return pd.DataFrame(recs, columns=["model", "end", "rank", "species", "domain", "rating"])


def build_all(long, tax, figdir, title="", struct="proteinmpnn_score", seq="protgpt2_score",
              violin_points=False, heatmap_value="z", rep_models=REP_MODELS, top_n=3,
              gap_models=FIG2B_MODELS, order_gap_by_delta=False,
              order_rep_by_delta=True, gap_seed=0):
    """Write all panels (HTML + PDF) + a top/bottom-species CSV into figdir.

    Emits the full 14-model violin *and* a readable representative-subset violin
    (``rep_models``). Call this for the full-14-model run."""
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    tb = top_bottom_species(long, n=top_n)
    tb.to_csv(figdir / "elo_top_bottom_species.csv", index=False)
    species_bars(long, figdir / "fig2a_species_bars.html", struct, seq, title)
    gap_dotplot(long, figdir / "fig2b_gap_dotplot.html", title,
                models=gap_models, order_by_delta=order_gap_by_delta,
                seed=gap_seed)
    domain_violin(long, figdir / "s1a_domain_violins.html", title, points=violin_points)
    domain_violin(long, figdir / "s1a_domain_violins_reps.html", title,
                  points=violin_points, models=rep_models,
                  order_by_delta=order_rep_by_delta)
    if tax is not None:
        phylum_model_heatmap(long, tax, figdir / "s1b_phylum_model_heatmap.html",
                             title, value=heatmap_value)
    print(f"  [paper figures] wrote 2A/2B/S1A(+reps)/S1B + top{top_n}-species CSV to {figdir}",
          flush=True)
