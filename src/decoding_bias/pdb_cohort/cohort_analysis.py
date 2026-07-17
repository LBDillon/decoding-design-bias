"""decoding_bias.pdb_cohort.cohort_analysis -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - run_cohort_elo
  - run_replication_stats
  - compare_cohort_to_main
  - run_matched_af2_control
"""

import importlib.util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_cohort_vd as RC
import score_variance_decomposition as svd
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from scipy import stats

# ---------- from run_cohort_elo.py ----------
run_cohort_elo_REPO = Path(__file__).resolve().parent.parent
run_cohort_elo_ELO_DIR = run_cohort_elo_REPO / 'paper_code' / '02_elo'
spec = importlib.util.spec_from_file_location('elor', run_cohort_elo_ELO_DIR / 'elo_rating.py')
run_cohort_elo_E = importlib.util.module_from_spec(spec)
COHORT = run_cohort_elo_REPO / 'design' / 'outputs' / 'independent_cohort' / 'cohort_pdb_scored.csv'
OUTDIR = run_cohort_elo_REPO / 'design' / 'outputs' / 'independent_cohort' / 'elo_cohort_unweighted'
CANDIDATES = ['proteinmpnn_score', 'solublempnn_score', 'caliby_score', 'soluble_caliby_score', 'esmif_score', 'mif_score', 'mifst_score', 'esm3_struct_cond_score', 'esm3_seq_only_score', 'ESM2_15B_pppl_score', 'esmc_6b_score', 'carp_640M_score', 'progen2_XL_score', 'protgpt2_score', 'AlkalineMPNN_score', 'AcidophileMPNN_score']
DOM = ['Archaea', 'Bacteria', 'Eukaryota']
def run_cohort_elo_main():
    df = pd.read_csv(COHORT)
    if 'avg_plddt' not in df.columns:
        df['avg_plddt'] = np.nan
    present = [c for c in CANDIDATES if c in df.columns and df[c].notna().sum() >= 100]
    print(f'cohort Elo on {len(present)} models: {present}')
    run_cohort_elo_E.run_full_elo_analysis(df, str(OUTDIR), score_columns=present, n_permutations=50, protein_column='protein_family', species_column='species', use_plddt_weighting=False)
    long = pd.read_csv(OUTDIR / 'results' / 'all_models_species_ratings_long.csv')
    g = long.groupby('model').apply(lambda x: pd.Series({'Archaea': x[x.domain == 'Archaea'].rating.mean(), 'Bacteria': x[x.domain == 'Bacteria'].rating.mean(), 'Eukaryota': x[x.domain == 'Eukaryota'].rating.mean(), 'Arch_minus_Euk': x[x.domain == 'Archaea'].rating.mean() - x[x.domain == 'Eukaryota'].rating.mean(), 'top_domain': x.groupby('domain').rating.mean().idxmax()})).reset_index()
    g = g.sort_values('Arch_minus_Euk', ascending=False)
    pd.set_option('display.width', 200)
    print('\n=== Cohort species Elo by domain (unweighted) ===')
    print(g.round(0).to_string(index=False))
    g.to_csv(OUTDIR / 'cohort_elo_domain_summary.csv', index=False)
    print(f"\nWrote {OUTDIR / 'cohort_elo_domain_summary.csv'}")
def run_cohort_elo__entry():
    spec.loader.exec_module(run_cohort_elo_E)
    run_cohort_elo_main()

# ---------- from run_replication_stats.py ----------
run_replication_stats_REPO = Path(__file__).resolve().parent.parent
IC = run_replication_stats_REPO / 'design' / 'outputs' / 'independent_cohort'
run_replication_stats_MAIN = run_replication_stats_REPO / 'dataset_update' / 'main_plus_r2_r3_analysis_v12_cli.csv'
RB = run_replication_stats_REPO / 'pdb_robustness'
run_replication_stats_PANEL = RC.MODELS
TYPE_C = {'structure': '#2E7D32', 'hybrid': '#F9A825', 'hybrid-ST': '#C0CA33', 'sequence': '#00838F', 'structure(FT)': '#7E57C2'}
def af2_full_residuals(models):
    """Per-model residual species on the full AFDB dataset (same method as the cohort)."""
    df = pd.read_csv(run_replication_stats_MAIN, low_memory=False)
    df = df[df.domain.isin(['Archaea', 'Bacteria', 'Eukaryota'])].copy()
    cc = df[df[RC.BIOPHYS].notna().all(axis=1)].copy()
    Z = ((cc[RC.BIOPHYS] - cc[RC.BIOPHYS].mean()) / cc[RC.BIOPHYS].std()).values
    (U, S, _) = np.linalg.svd(Z, full_matrices=False)
    (cc['PC1'], cc['PC2']) = (U[:, 0] * S[0], U[:, 1] * S[1])
    out = {}
    for m in models:
        (col, kind) = run_replication_stats_PANEL[m]
        if col not in cc.columns or cc[col].notna().sum() < RC.MIN_N:
            continue
        rec = RC.decompose_one(cc, col, kind)
        if rec:
            out[m] = rec
    return out
def corr_block(a, b, names, label):
    """Pearson + Spearman with p, on aligned vectors a,b across the model panel."""
    (a, b) = (np.asarray(a, float), np.asarray(b, float))
    (pr, pp) = stats.pearsonr(a, b)
    (sr, sp) = stats.spearmanr(a, b)
    return [dict(comparison=label, n_models=len(a), pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp)]
def repel_labels(ax, xs, ys, labels, fontsize=7, n_iter=150):
    """Dependency-free text repel: put each label by its point, then iteratively push
    overlapping label boxes apart (in display coords) and draw a thin leader line back
    to the marker. Avoids the overlapping model names in the replication scatter."""
    fig = ax.figure
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    texts = [ax.text(x, y, s, fontsize=fontsize, zorder=6, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75)) for (x, y, s) in zip(xs, ys, labels)]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    pos = pts + np.array([7.0, 7.0])
    inv = ax.transData.inverted()
    for _ in range(n_iter):
        for (t, p) in zip(texts, pos):
            t.set_position(inv.transform(p))
        fig.canvas.draw()
        bb = [t.get_window_extent(rend) for t in texts]
        moved = False
        for i in range(len(texts)):
            push = np.zeros(2)
            for j in range(len(texts)):
                if i != j and bb[i].overlaps(bb[j]):
                    ci = np.array([(bb[i].x0 + bb[i].x1) / 2, (bb[i].y0 + bb[i].y1) / 2])
                    cj = np.array([(bb[j].x0 + bb[j].x1) / 2, (bb[j].y0 + bb[j].y1) / 2])
                    diff = ci - cj
                    norm = np.hypot(*diff) or 1.0
                    push += diff / norm * 3.0
            d0 = pos[i] - pts[i]
            n0 = np.hypot(*d0) or 1.0
            if n0 < 12:
                push += d0 / n0 * 2.0
            if push.any():
                pos[i] += push
                moved = True
        if not moved:
            break
    for (t, p0) in zip(texts, pts):
        (x, y) = t.get_position()
        (x0, y0) = inv.transform(p0)
        ax.plot([x0, x], [y0, y], color='0.6', lw=0.4, zorder=5)
def run_replication_stats_main():
    cohort = pd.read_csv(IC / 'cohort_score_variance_decomposition.csv').set_index('model')
    af2m = pd.read_csv(IC / 'matched_af2_vd_control.csv').set_index('model')
    models = [m for m in run_replication_stats_PANEL if m in cohort.index and m in af2m.index]
    af2full = af2_full_residuals(models)
    rows = []
    for m in models:
        (col, kind) = run_replication_stats_PANEL[m]
        rec = dict(model=m, type=kind, PDB_resid_raw=cohort.loc[m, 'dSpecies_given_family_biophys'], PDB_resid_adj=cohort.loc[m, 'residual_species_R2_adj'], PDB_retention=cohort.loc[m, 'species_effect_retention_given_family_biophys'], PDB_species_p=cohort.loc[m, 'species_p_given_family_biophys'], AF2m_resid_raw_mean=af2m.loc[m, 'AF2_resid_raw_mean'], AF2m_resid_raw_sd=af2m.loc[m, 'AF2_resid_raw_sd'], AF2m_resid_adj_mean=af2m.loc[m, 'AF2_resid_adj_mean'], AF2m_retention_mean=af2m.loc[m, 'AF2_retention_mean'])
        if m in af2full:
            rec['AF2full_resid_raw'] = af2full[m]['dSpecies_given_family_biophys']
            rec['AF2full_resid_adj'] = af2full[m]['residual_species_R2_adj']
            rec['AF2full_retention'] = af2full[m]['species_effect_retention_given_family_biophys']
            rec['AF2full_species_p'] = af2full[m]['species_p_given_family_biophys']
        sd = rec['AF2m_resid_raw_sd'] or np.nan
        z = (rec['PDB_resid_raw'] - rec['AF2m_resid_raw_mean']) / sd
        rec['departure_z'] = z
        rec['departure_p'] = 2 * stats.norm.sf(abs(z))
        rows.append(rec)
    res = pd.DataFrame(rows)
    order = res['departure_p'].rank(method='first')
    m_n = res['departure_p'].notna().sum()
    res['departure_p_BH'] = (res['departure_p'] * m_n / order).clip(upper=1.0)
    res.to_csv(RB / 'data' / 'vd_replication_stats.csv', index=False)
    names = res['model'].tolist()
    cors = []
    cors += corr_block(res['PDB_resid_raw'], res['AF2m_resid_raw_mean'], names, 'PDB vs AF2-matched (raw, same N=876)')
    have_full = res['AF2full_resid_adj'].notna()
    cors += corr_block(res.loc[have_full, 'PDB_resid_adj'], res.loc[have_full, 'AF2full_resid_adj'], res.loc[have_full, 'model'].tolist(), 'PDB vs AF2-full (adjusted)')
    cors += corr_block(res.loc[have_full, 'AF2m_resid_adj_mean'], res.loc[have_full, 'AF2full_resid_adj'], res.loc[have_full, 'model'].tolist(), 'AF2-matched vs AF2-full (adjusted)')
    cors += corr_block(res['PDB_retention'], res['AF2m_retention_mean'], names, 'PDB vs AF2-matched (retention)')
    cors = pd.DataFrame(cors)
    cors.to_csv(RB / 'data' / 'vd_replication_correlations.csv', index=False)
    res['class'] = res['type'].map(lambda t: 'sequence' if t == 'sequence' else 'structure/hybrid')
    paired = []
    for (cls, g) in res.groupby('class'):
        diff = (g['PDB_resid_raw'] - g['AF2m_resid_raw_mean']).values
        try:
            (w, p) = stats.wilcoxon(diff)
        except ValueError:
            (w, p) = (np.nan, np.nan)
        paired.append(dict(model_class=cls, n=len(g), median_PDB_minus_AF2m=float(np.median(diff)), wilcoxon_W=w, wilcoxon_p=p))
    paired = pd.DataFrame(paired)
    pd.set_option('display.width', 200)
    print('=== Per-model residual species across inputs (+ departure of PDB from matched AF2) ===')
    show = ['model', 'type', 'PDB_resid_raw', 'AF2m_resid_raw_mean', 'AF2full_resid_adj', 'departure_z', 'departure_p_BH']
    print(res[show].round(3).to_string(index=False))
    print('\n=== (1) Replication correlations across the model panel ===')
    print(cors.round(3).to_string(index=False))
    print('\n=== (3) Paired PDB-minus-AF2matched within model class (Wilcoxon) ===')
    print(paired.round(4).to_string(index=False))

    def fnum(x):
        return '--' if pd.isna(x) else f'{x:.3f}'

    def fp(x):
        return '--' if pd.isna(x) else '$<$0.001' if x < 0.001 else f'{x:.3f}'
    lines = []
    for (_, r) in cors.iterrows():
        lines.append(f"{r['comparison']} & {int(r['n_models'])} & {fnum(r['pearson_r'])} & {fp(r['pearson_p'])} & {fnum(r['spearman_rho'])} & {fp(r['spearman_p'])} \\\\")
    tex = '% auto-generated by design/run_replication_stats.py\n\\begin{table}[ht]\\centering\\small\n\\caption{Replication of the per-model residual taxonomic bias across inputs: correlation of the per-model residual-species vector between the experimental-PDB cohort, the matched AFDB subsample, and the full AFDB dataset. Spearman (rank) tests whether the ordering of models by bias replicates; raw residuals are used where $N$ is matched, adjusted $R^2$ across different $N$.}\\label{tab:vd-replication}\n\\begin{tabular}{lrrrrr}\n\\toprule\nComparison & $k$ & Pearson $r$ & $p$ & Spearman $\\rho$ & $p$ \\\\\n\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (RB / 'tables' / 'table_vd_replication.tex').write_text(tex)
    (fig, ax) = plt.subplots(1, 2, figsize=(13, 6))
    for (a, (xc, yc, xl, yl, ttl, sub)) in zip(ax, [('AF2m_resid_raw_mean', 'PDB_resid_raw', 'AF2-matched residual species (raw)', 'PDB cohort residual species (raw)', 'Same N (876): PDB vs matched AFDB', cors[cors.comparison.str.startswith('PDB vs AF2-matched (raw')]), ('AF2full_resid_adj', 'PDB_resid_adj', 'AF2-full residual species (adjusted)', 'PDB cohort residual species (adjusted)', 'Across N: PDB vs full AFDB', cors[cors.comparison.str.startswith('PDB vs AF2-full')])]):
        d = res.dropna(subset=[xc, yc])
        a.scatter(d[xc], d[yc], c=[TYPE_C.get(t, '#666') for t in d['type']], s=70, zorder=3)
        lo = min(d[xc].min(), d[yc].min())
        hi = max(d[xc].max(), d[yc].max())
        pad = (hi - lo) * 0.08
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.8, alpha=0.5, label='y = x')
        a.set_xlim(lo - pad, hi + pad)
        a.set_ylim(lo - pad, hi + pad)
        repel_labels(a, d[xc].values, d[yc].values, d['model'].tolist())
        sr = sub['spearman_rho'].iloc[0]
        sp = sub['spearman_p'].iloc[0]
        pr = sub['pearson_r'].iloc[0]
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.set_title(ttl)
        a.text(0.04, 0.96, f'Spearman ρ={sr:.2f} (p={sp:.3f})\nPearson r={pr:.2f}', transform=a.transAxes, va='top', fontsize=10, bbox=dict(boxstyle='round', fc='white', ec='0.7'))
        a.legend(loc='lower right', fontsize=8)
        a.grid(alpha=0.2)
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=c, label=t) for (t, c) in TYPE_C.items() if t in set(res['type'])]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle('Does the per-model residual taxonomic bias replicate across inputs?', fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(RB / 'figures' / 'fig_vd_replication.png', dpi=150)
    fig.savefig(RB / 'figures' / 'fig_vd_replication.pdf')
    plt.close(fig)
    print('\nWrote pdb_robustness/data/vd_replication_stats.csv, vd_replication_correlations.csv')
    print('Wrote pdb_robustness/tables/table_vd_replication.tex, figures/fig_vd_replication.{png,pdf}')
def run_replication_stats__entry():
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(run_replication_stats_REPO / 'design'))
    (RB / 'data').mkdir(parents=True, exist_ok=True)
    (RB / 'tables').mkdir(exist_ok=True)
    (RB / 'figures').mkdir(exist_ok=True)
    run_replication_stats_main()

# ---------- from compare_cohort_to_main.py ----------
HERE = Path(__file__).resolve().parent
COH = HERE / 'outputs' / 'independent_cohort'
MAIN_META = HERE.parent / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
DOMAIN_ORDER = ['Eukaryota', 'Bacteria', 'Archaea']
(C_MAIN, C_COH) = ('#55A868', '#4C72B0')
def load():
    main = pd.read_csv(MAIN_META, low_memory=False)
    coh = pd.read_csv(COH / 'cohort_pdb_scoring_inputs.csv')
    return (main, coh)
def summary_table(main, coh):

    def length_of(df):
        if 'Length' in df:
            return pd.to_numeric(df['Length'], errors='coerce')
        return pd.to_numeric(df['resolved_len'], errors='coerce')
    rows = []
    rows.append(('N proteins', f'{len(main):,}', f'{len(coh):,}'))
    rows.append(('Structure source', 'AlphaFold2 model', 'Experimental X-ray'))
    rows.append(('Sequence scored', 'Full UniProt', 'Resolved chain'))
    for d in DOMAIN_ORDER:
        mp = (main['domain'] == d).mean() * 100
        cp = (coh['domain'] == d).mean() * 100
        rows.append((f'{d} (\\%)', f'{mp:.1f}', f'{cp:.1f}'))
    rows.append(('Distinct families', f"{main['protein_family'].nunique():,}", f"{coh['protein_family'].nunique():,}"))
    sp_main = main['species'].nunique() if 'species' in main else np.nan
    sp_coh = coh['species_collapsed'].nunique()
    rows.append(('Distinct species', f'{sp_main:,}', f'{sp_coh:,}'))
    rows.append(('Ribosomal (\\%)', f"{(main['broad_function'] == 'ribosomal').mean() * 100:.1f}", f"{(coh['broad_function'] == 'ribosomal').mean() * 100:.1f}"))
    (lm, lc) = (length_of(main), length_of(coh))
    rows.append(('Median length (aa)', f'{lm.median():.0f}', f'{lc.median():.0f}'))
    if 'resolution_A' in coh:
        rows.append(('Median resolution (\\AA)', '--', f"{coh['resolution_A'].median():.2f}"))
    return pd.DataFrame(rows, columns=['Property', 'Main dataset', 'Independent PDB cohort'])
def write_tex(tab):
    body = '\n'.join((f'{r.Property} & {r._1} & {r._2} \\\\' for r in tab.itertuples(index=False, name='R')))
    tex = '\\begin{table}[t]\n\\centering\n\\caption{Comparison of the published main dataset and the independent experimental-PDB replication cohort.}\n\\label{tab:cohort_vs_main}\n\\begin{tabular}{lrr}\n\\hline\nProperty & Main dataset & Independent PDB cohort \\\\\n\\hline\n' + body + '\n\\hline\n\\end{tabular}\n\\end{table}\n'
    (COH / 'cohort_vs_main_table.tex').write_text(tex)
def fig(main, coh):
    (fig, ax) = plt.subplots(2, 2, figsize=(11, 8))
    mp = [(main['domain'] == d).mean() * 100 for d in DOMAIN_ORDER]
    cp = [(coh['domain'] == d).mean() * 100 for d in DOMAIN_ORDER]
    x = np.arange(len(DOMAIN_ORDER))
    w = 0.38
    ax[0, 0].bar(x - w / 2, mp, w, label='Main', color=C_MAIN)
    ax[0, 0].bar(x + w / 2, cp, w, label='PDB cohort', color=C_COH)
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(DOMAIN_ORDER)
    ax[0, 0].set_ylabel('% of proteins')
    ax[0, 0].set_title('(a) Taxonomic domain')
    ax[0, 0].legend(frameon=False)
    order = main['broad_function'].value_counts().head(10).index.tolist()
    for c in coh['broad_function'].value_counts().head(10).index:
        if c not in order:
            order.append(c)
    order = order[:12]
    mf = [(main['broad_function'] == c).mean() * 100 for c in order]
    cf = [(coh['broad_function'] == c).mean() * 100 for c in order]
    y = np.arange(len(order))
    ax[0, 1].barh(y - w / 2, mf, w, label='Main', color=C_MAIN)
    ax[0, 1].barh(y + w / 2, cf, w, label='PDB cohort', color=C_COH)
    ax[0, 1].set_yticks(y)
    ax[0, 1].set_yticklabels(order, fontsize=8)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_xlabel('% of proteins')
    ax[0, 1].set_title('(b) Broad function')
    ax[0, 1].legend(frameon=False)
    lm = pd.to_numeric(main['Length'], errors='coerce').dropna()
    lc = pd.to_numeric(coh['resolved_len'], errors='coerce').dropna()
    bins = np.linspace(0, 1000, 41)
    ax[1, 0].hist(lm, bins=bins, density=True, alpha=0.6, color=C_MAIN, label='Main (UniProt len)')
    ax[1, 0].hist(lc, bins=bins, density=True, alpha=0.6, color=C_COH, label='PDB cohort (resolved len)')
    ax[1, 0].set_xlabel('Length (aa)')
    ax[1, 0].set_ylabel('Density')
    ax[1, 0].set_title('(c) Sequence length')
    ax[1, 0].legend(frameon=False)
    res = pd.to_numeric(coh['resolution_A'], errors='coerce').dropna()
    ax[1, 1].hist(res, bins=np.linspace(0.8, 2.6, 28), color=C_COH, alpha=0.85)
    ax[1, 1].axvline(res.median(), color='k', ls='--', lw=1, label=f'median {res.median():.2f} Å')
    ax[1, 1].set_xlabel('Resolution (Å)')
    ax[1, 1].set_ylabel('Proteins')
    ax[1, 1].set_title('(d) PDB cohort resolution')
    ax[1, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(COH / 'cohort_vs_main_figure.png', dpi=200)
    fig.savefig(COH / 'cohort_vs_main_figure.pdf')
def compare_cohort_to_main_main():
    (m, c) = load()
    tab = summary_table(m, c)
    tab.to_csv(COH / 'cohort_vs_main_table.csv', index=False)
    write_tex(tab)
    fig(m, c)
    print(tab.to_string(index=False))
    print(f'\nwrote cohort_vs_main_table.csv/.tex and cohort_vs_main_figure.png/.pdf')
def compare_cohort_to_main__entry():
    matplotlib.use('Agg')
    compare_cohort_to_main_main()

# ---------- from run_matched_af2_control.py ----------
run_matched_af2_control_REPO = Path(__file__).resolve().parent.parent
SVD_DIR = run_matched_af2_control_REPO / 'paper_code' / '03_variance_decomposition'
run_matched_af2_control_ELO_DIR = run_matched_af2_control_REPO / 'paper_code' / '02_elo'
_spec = importlib.util.spec_from_file_location('elor', run_matched_af2_control_ELO_DIR / 'elo_rating.py')
run_matched_af2_control_E = importlib.util.module_from_spec(_spec)
run_matched_af2_control_MAIN = run_matched_af2_control_REPO / 'dataset_update' / 'main_plus_r2_r3_analysis_v12_cli.csv'
COH_VD = run_matched_af2_control_REPO / 'design' / 'outputs' / 'independent_cohort' / 'cohort_score_variance_decomposition.csv'
COH_ELO = run_matched_af2_control_REPO / 'design' / 'outputs' / 'independent_cohort' / 'elo_cohort_unweighted' / 'cohort_elo_domain_summary.csv'
OUT = run_matched_af2_control_REPO / 'design' / 'outputs' / 'independent_cohort'
SEQ = [f for f in svd.SEQ if f not in ('charge_at_ph7', 'small_residue_fraction')]
BIOPHYS = SEQ + svd.STRUCT
run_matched_af2_control_PANEL = {'ProteinMPNN': ('proteinmpnn_score', 'structure'), 'SolubleMPNN': ('solublempnn_score', 'structure'), 'Caliby': ('caliby_score', 'structure'), 'SolubleCaliby': ('soluble_caliby_score', 'structure'), 'ESM-IF': ('esmif_score', 'structure'), 'ESM3-struct': ('esm3_struct_cond_score', 'structure'), 'MIF': ('mif_score', 'hybrid'), 'MIF-ST': ('mifst_score', 'hybrid-ST'), 'ESM3-seq': ('esm3_seq_only_score', 'sequence'), 'ESM2-15B': ('ESM2_15B_pppl_score', 'sequence'), 'CARP-640M': ('carp_640M_score', 'sequence'), 'ProGen2-XL': ('progen2_XL_score', 'sequence'), 'ProtGPT2': ('protgpt2_score', 'sequence'), 'ESMC-6B': ('esmc_6b_score', 'sequence')}
TARGET = {'Eukaryota': 396, 'Bacteria': 364, 'Archaea': 116}
B_VD = 30
B_ELO = 8
def cohort_models():
    """Models the cohort VD produced AND that exist in the AFDB table (so the matched
    control can be computed). A cohort-only model - e.g. ESMC scored only on the PDB
    chains - is skipped here until it is also scored on the main AFDB dataset."""
    cv = pd.read_csv(COH_VD)
    main_cols = set(pd.read_csv(run_matched_af2_control_MAIN, nrows=1).columns)
    return [m for m in cv['model'] if m in run_matched_af2_control_PANEL and run_matched_af2_control_PANEL[m][0] in main_cols]
def subsample(pool, seed):
    parts = [pool[pool.domain == d].sample(n=min(n, int((pool.domain == d).sum())), random_state=seed) for (d, n) in TARGET.items()]
    return pd.concat(parts).reset_index(drop=True)
def make_bases(sub, score_cols):
    cc = sub.dropna(subset=BIOPHYS + ['protein_family', 'species'] + score_cols).copy()
    Z = ((cc[BIOPHYS] - cc[BIOPHYS].mean()) / cc[BIOPHYS].std()).values
    (U, S, _) = np.linalg.svd(Z, full_matrices=False)
    PC = U[:, :2] * S[:2]
    fam = pd.get_dummies(cc['protein_family'], drop_first=True).values.astype(float)
    spc = pd.get_dummies(cc['species'], drop_first=True).values.astype(float)
    (sp_codes, sp_levels) = pd.factorize(cc['species'])
    bases = {'Family': svd.make_Q(fam), 'Biophys+Family': svd.make_Q(np.hstack([Z, fam])), 'Full': svd.make_Q(np.hstack([Z, fam, spc]))}
    return (cc, bases, sp_codes, len(sp_levels))
def resid_metrics(y, bases, sp_codes, n_sp):
    R = {k: svd.r2_from_Q(y, Q, p) for (k, (Q, p)) in bases.items()}
    d_raw = R['Full'][0] - R['Biophys+Family'][0]
    d_adj = R['Full'][1] - R['Biophys+Family'][1]
    coll = svd.species_effect_collapse(y, sp_codes, n_sp, bases)
    return (d_raw, d_adj, coll['species_effect_retention_given_family_biophys'])
def vd_control(models):
    pool = pd.read_csv(run_matched_af2_control_MAIN, low_memory=False)
    pool = pool[pool.domain.isin(TARGET)]
    pool = pool[~pool['broad_function'].astype(str).str.contains('ribosom', case=False)]
    cols = [run_matched_af2_control_PANEL[m][0] for m in models]
    print(f'AF2 pool (no ribosomal): {len(pool)}  | matched subsamples N={sum(TARGET.values())} x {B_VD}')
    acc = {m: {'raw': [], 'adj': [], 'ret': []} for m in models}
    for b in range(B_VD):
        sub = subsample(pool, seed=1000 + b)
        (cc, bases, sp_codes, n_sp) = make_bases(sub, cols)
        for m in models:
            col = run_matched_af2_control_PANEL[m][0]
            y = cc[col].values.astype(float)
            y = (y - y.mean()) / y.std()
            (d_raw, d_adj, ret) = resid_metrics(y, bases, sp_codes, n_sp)
            acc[m]['raw'].append(d_raw)
            acc[m]['adj'].append(d_adj)
            acc[m]['ret'].append(ret)
    cv = pd.read_csv(COH_VD).set_index('model')
    rows = []
    for m in models:
        a = acc[m]
        rows.append(dict(model=m, type=run_matched_af2_control_PANEL[m][1], PDB_resid_raw=cv.loc[m, 'dSpecies_given_family_biophys'], AF2_resid_raw_mean=np.mean(a['raw']), AF2_resid_raw_sd=np.std(a['raw']), PDB_resid_adj=cv.loc[m, 'residual_species_R2_adj'], AF2_resid_adj_mean=np.mean(a['adj']), AF2_resid_adj_sd=np.std(a['adj']), PDB_retention=cv.loc[m, 'species_effect_retention_given_family_biophys'], AF2_retention_mean=np.mean(a['ret']), AF2_retention_sd=np.std(a['ret'])))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'matched_af2_vd_control.csv', index=False)
    pd.set_option('display.width', 220)
    print('\n=== VD: PDB cohort vs AF2-matched (N & composition matched) ===')
    print(res.round(3).to_string(index=False))
    print(f"Wrote {OUT / 'matched_af2_vd_control.csv'}")
    return res
def elo_control(models):
    pool = pd.read_csv(run_matched_af2_control_MAIN, low_memory=False)
    pool = pool[pool.domain.isin(TARGET)]
    pool = pool[~pool['broad_function'].astype(str).str.contains('ribosom', case=False)]
    if 'avg_plddt' not in pool.columns:
        pool['avg_plddt'] = np.nan
    cols = [run_matched_af2_control_PANEL[m][0] for m in models]
    gaps = {m: [] for m in models}
    tmp = Path(tempfile.mkdtemp())
    for b in range(B_ELO):
        sub = subsample(pool, seed=2000 + b)
        outdir = tmp / f'rep{b}'
        run_matched_af2_control_E.run_full_elo_analysis(sub, str(outdir), score_columns=cols, n_permutations=30, protein_column='protein_family', species_column='species', use_plddt_weighting=False)
        long = pd.read_csv(outdir / 'results' / 'all_models_species_ratings_long.csv')
        g = long.groupby('model').apply(lambda x: x[x.domain == 'Archaea'].rating.mean() - x[x.domain == 'Eukaryota'].rating.mean())
        col2name = {v[0]: k for (k, v) in run_matched_af2_control_PANEL.items()}
        for (col, val) in g.items():
            name = col2name.get(col, col)
            if name in gaps:
                gaps[name].append(val)
        print(f'  elo rep {b + 1}/{B_ELO} done')
    shutil.rmtree(tmp, ignore_errors=True)
    coh = pd.read_csv(COH_ELO)
    col2name = {v[0]: k for (k, v) in run_matched_af2_control_PANEL.items()}
    coh['name'] = coh['model'].map(lambda c: col2name.get(c, c))
    coh_gap = coh.set_index('name')['Arch_minus_Euk']
    rows = []
    for m in models:
        if not gaps[m]:
            continue
        rows.append(dict(model=m, type=run_matched_af2_control_PANEL[m][1], PDB_arch_minus_euk=coh_gap.get(m, np.nan), AF2_arch_minus_euk_mean=np.mean(gaps[m]), AF2_arch_minus_euk_sd=np.std(gaps[m])))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'matched_af2_elo_control.csv', index=False)
    print('\n=== Elo Arch−Euk gap: PDB cohort vs AF2-matched ===')
    print(res.round(0).to_string(index=False))
    print(f"Wrote {OUT / 'matched_af2_elo_control.csv'}")
    return res
def run_matched_af2_control__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(SVD_DIR))
    _spec.loader.exec_module(run_matched_af2_control_E)
    models = cohort_models()
    print(f'Matching {len(models)} cohort-present models: {models}\n')
    vd_control(models)
    elo_control(models)

_STEPS = {
    'run-cohort-elo': run_cohort_elo__entry,
    'run-replication-stats': run_replication_stats__entry,
    'compare-cohort-to-main': compare_cohort_to_main__entry,
    'run-matched-af2-control': run_matched_af2_control__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

