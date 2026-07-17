"""decoding_bias.design.design_tm_tables -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - predict_tm_full
  - tm_shift_analysis
  - shift_significance_tables
"""

from __future__ import annotations

import io
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests
import sys
import time
import warnings
from pathlib import Path
from scipy import stats
from features_for_designs import MIXED_FEATURES

# ---------- from predict_tm_full.py ----------
predict_tm_full_HERE = Path(__file__).resolve().parent
META = predict_tm_full_HERE.parent / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
predict_tm_full_OUT = predict_tm_full_HERE / 'outputs' / 'dataset_tm_predictions.csv'
BASE = os.environ.get('DEEPSTABP_URL', 'http://localhost:8000').rstrip('/')
GROWTH_TEMP = int(os.environ.get('GROWTH_TEMP', '37'))
MT_MODE = os.environ.get('MT_MODE', 'Lysate')
BATCH = int(os.environ.get('BATCH', '100'))
STD = set('ACDEFGHIKLMNPQRSTVWY')
def clean(seq):
    return ''.join((c if c in STD else 'X' for c in str(seq).upper()))
def parse_prediction(pred):
    """Coerce the {'Prediction': ...} payload to a DataFrame with Protein, Tm."""
    if isinstance(pred, str):
        try:
            df = pd.read_json(io.StringIO(pred))
        except ValueError:
            df = pd.DataFrame(json.loads(pred))
    else:
        df = pd.DataFrame(pred)
    cols = {c.lower(): c for c in df.columns}
    pcol = cols.get('protein') or cols.get('header') or list(df.columns)[0]
    tcol = cols.get('tm') or [c for c in df.columns if c.lower().startswith('tm')][0]
    return df[[pcol, tcol]].rename(columns={pcol: 'Protein', tcol: 'Tm'})
def predict_tm_full_main():
    m = pd.read_csv(META, low_memory=False).dropna(subset=['sequence'])
    m = m[m.sequence.str.len() > 0][['Entry', 'sequence']]
    done = set()
    if predict_tm_full_OUT.exists():
        done = set(pd.read_csv(predict_tm_full_OUT).Entry.astype(str))
        print(f'resuming: {len(done)} already predicted')
    todo = m[~m.Entry.astype(str).isin(done)].reset_index(drop=True)
    print(f'{len(todo)} to predict via {BASE}/api/v1/predict (growth_temp={GROWTH_TEMP}, mt_mode={MT_MODE}, batch={BATCH})')
    predict_tm_full_OUT.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not predict_tm_full_OUT.exists()
    for i in range(0, len(todo), BATCH):
        sub = todo.iloc[i:i + BATCH]
        payload = {'growth_temp': GROWTH_TEMP, 'mt_mode': MT_MODE, 'fasta': [{'header': r.Entry, 'sequence': clean(r.sequence)} for r in sub.itertuples()]}
        for attempt in range(4):
            try:
                r = requests.post(f'{BASE}/api/v1/predict', json=payload, timeout=600)
                r.raise_for_status()
                df = parse_prediction(r.json()['Prediction'])
                df['Entry'] = df['Protein'].astype(str)
                df[['Entry', 'Tm']].to_csv(predict_tm_full_OUT, mode='a', header=header_needed, index=False)
                header_needed = False
                print(f'  batch {i // BATCH + 1}/{-(-len(todo) // BATCH)}: +{len(df)}  (Tm {df.Tm.min():.1f}-{df.Tm.max():.1f})', flush=True)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f'  batch {i // BATCH + 1} attempt {attempt + 1} failed: {str(e)[:80]} - retry in {wait}s', flush=True)
                time.sleep(wait)
        else:
            print('  giving up on this batch; re-run to resume.')
            sys.exit(1)
    res = pd.read_csv(predict_tm_full_OUT)
    print(f'\nDone: {len(res)} Tm predictions written to {predict_tm_full_OUT}')
    print(res.Tm.describe().round(1).to_string())
def predict_tm_full__entry():
    predict_tm_full_main()

# ---------- from tm_shift_analysis.py ----------
tm_shift_analysis_HERE = Path(__file__).resolve().parent
tm_shift_analysis_REPO = tm_shift_analysis_HERE.parents[1]
TM = tm_shift_analysis_REPO / 'design' / 'Design_WT_TM.csv'
INP = tm_shift_analysis_REPO / 'design' / 'design_input_proteins.csv'
tm_shift_analysis_OUT = tm_shift_analysis_REPO / 'design' / 'outputs'
def parse_id(pid):
    if pid.endswith('__WT'):
        return (pid[:-4], 'WT', -1)
    m = re.match('(.+)__(.+)__s(\\d+)$', pid)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (pid, '?', -1)
def tm_shift_analysis_main():
    t = pd.read_csv(TM)
    t[['uniprot_id', 'model', 'sample_idx']] = pd.DataFrame([parse_id(p) for p in t.protein_id], index=t.index)
    meta = pd.read_csv(INP)[['uniprot_id', 'domain', 'rank_class', 'species']]
    t = t.merge(meta, on='uniprot_id', how='left')
    wt = t[t.model == 'WT'].set_index('uniprot_id')['Tm']
    des = t[t.model != 'WT'].copy()
    des['wt_Tm'] = des.uniprot_id.map(wt)
    des['dTm'] = des.Tm - des.wt_Tm
    pp = des.groupby(['model', 'uniprot_id', 'domain']).agg(dTm=('dTm', 'mean'), design_Tm=('Tm', 'mean'), wt_Tm=('wt_Tm', 'first')).reset_index()
    pp.to_csv(tm_shift_analysis_OUT / 'tm_shift_per_protein.csv', index=False)
    rows = []
    for (model, g) in pp.groupby('model'):
        x = g.dTm.values
        dz = x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan
        p = stats.wilcoxon(x).pvalue if len(x) >= 3 else np.nan
        rows.append(dict(model=model, mean_dTm=x.mean(), median_dTm=np.median(x), sd_dTm=x.std(ddof=1), cohens_dz=dz, wilcoxon_p=p, n=len(x)))
    res = pd.DataFrame(rows)
    from scipy.stats import false_discovery_control
    res['p_fdr'] = false_discovery_control(res.wilcoxon_p.fillna(1))
    res = res.sort_values('mean_dTm', ascending=False)
    res.to_csv(tm_shift_analysis_OUT / 'tm_shift_by_model.csv', index=False)
    print('=== WT→design ΔTm (°C) per model (DeepStabP) ===')
    print(res[['model', 'mean_dTm', 'median_dTm', 'cohens_dz', 'p_fdr', 'n']].round(2).to_string(index=False))
    dom = pp.groupby(['model', 'domain']).dTm.agg(['mean', 'median', 'count']).round(2).reset_index()
    dom.to_csv(tm_shift_analysis_OUT / 'tm_shift_by_model_domain.csv', index=False)
    print('\n=== mean ΔTm by model × domain ===')
    print(dom.pivot(index='model', columns='domain', values='mean').round(1).to_string())
    (fig, ax) = plt.subplots(figsize=(11, 6))
    order = res.model.tolist()
    dom_cols = {'Archaea': '#F57C00', 'Bacteria': '#1976D2', 'Eukaryota': '#388E3C'}
    for (i, m) in enumerate(order):
        g = pp[pp.model == m]
        ax.boxplot(g.dTm, positions=[i], widths=0.6, showfliers=False, medianprops=dict(color='black'))
        for (dmn, gg) in g.groupby('domain'):
            ax.scatter(np.full(len(gg), i) + np.random.uniform(-0.18, 0.18, len(gg)), gg.dTm, c=dom_cols.get(dmn, 'grey'), s=22, alpha=0.8, label=dmn)
    ax.axhline(0, color='red', ls='--', lw=1, alpha=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel('ΔTm (design − WT), °C')
    ax.set_title('Predicted thermostability shift per model (per-protein means)')
    (h, l) = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), title='domain', fontsize=8)
    fig.tight_layout()
    fig.savefig(tm_shift_analysis_OUT / 'tm_shift_distributions.png', dpi=150)
    print('\nwrote tm_shift_distributions.png, tm_shift_by_model.csv, tm_shift_per_protein.csv')
def tm_shift_analysis__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(tm_shift_analysis_HERE))
    matplotlib.use('Agg')
    tm_shift_analysis_OUT.mkdir(exist_ok=True)
    tm_shift_analysis_main()

# ---------- from shift_significance_tables.py ----------
shift_significance_tables_HERE = Path(__file__).resolve().parent
shift_significance_tables_REPO = shift_significance_tables_HERE.parents[1]
shift_significance_tables_OUT = shift_significance_tables_REPO / 'design' / 'outputs'
DEGENERATE = ['sequence_length', 'surface_exposure']
PROPERTIES = [p for p in MIXED_FEATURES if p not in DEGENERATE]
T_CRIT_7 = stats.t.ppf(0.975, df=7)
def build_master(designs: pd.DataFrame, wt: pd.DataFrame) -> pd.DataFrame:
    wt_idx = wt.set_index('uniprot_id')
    pop_sd = {p: wt[p].std(ddof=1) for p in PROPERTIES}
    rows = []
    for ((uid, model), g) in designs.groupby(['uniprot_id', 'model']):
        if uid not in wt_idx.index:
            continue
        domain = g['domain'].iloc[0]
        for p in PROPERTIES:
            vals = g[p].dropna().values
            wt_val = wt_idx.loc[uid, p]
            if len(vals) < 3 or not np.isfinite(wt_val):
                continue
            design_mean = vals.mean()
            shift = design_mean - wt_val
            sd_rep = vals.std(ddof=1)
            shifts = vals - wt_val
            if sd_rep > 0:
                (t, p_t) = stats.ttest_1samp(shifts, 0.0)
                dz = shift / sd_rep
                margin = T_CRIT_7 * sd_rep / np.sqrt(len(vals))
            else:
                (t, dz, margin) = (np.nan, 0.0 if shift == 0 else np.inf, 0.0)
                p_t = 1.0 if np.isclose(shift, 0.0) else np.nan
            try:
                (_, p_w) = stats.wilcoxon(shifts)
            except ValueError:
                p_w = np.nan
            rows.append(dict(Entry=uid, model=model, domain=domain, property=p, starting_value=round(float(wt_val), 4), design_value=round(float(design_mean), 4), mean_shift=round(float(shift), 4), shift_z=round(float(shift / pop_sd[p]), 4) if pop_sd[p] else np.nan, margin=round(float(margin), 4), d_z=round(float(dz), 3), p=p_t, p_wilcox=p_w, n=len(vals)))
    m = pd.DataFrame(rows)
    ok = m['p'].notna()
    m.loc[ok, 'p_fdr'] = _bh(m.loc[ok, 'p'].values)
    m['sig'] = m['p_fdr'] < 0.05
    return m
def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    return q
def _fmt(x, sig=False):
    """+0.21* style: signed value with a star when significant."""
    if pd.isna(x):
        return ''
    return f'{x:+.3f}' + ('*' if sig else '')
def shift_significance_tables_main():
    designs = pd.read_csv(shift_significance_tables_OUT / 'designs_features.csv')
    wt = pd.read_csv(shift_significance_tables_OUT / 'wt_features.csv')
    m = build_master(designs, wt)
    m.to_csv(shift_significance_tables_OUT / 'shift_significance_master.csv', index=False)
    example_protein = designs['uniprot_id'].iloc[0]
    pf = m[m.Entry == example_protein].copy()
    pf['cell'] = pf.apply(lambda r: f"{r.mean_shift:+.3f} +/-{r.margin:.3f}{('*' if r.sig else '')}", axis=1)
    pf_tbl = pf.pivot(index='property', columns='model', values='cell').reindex(PROPERTIES)
    pf_tbl.to_csv(shift_significance_tables_OUT / f'shift_table_protein_{example_protein}.csv')
    example_model = 'ProteinMPNN'
    mf = m[m.model == example_model].copy()
    mf['cell'] = mf.apply(lambda r: _fmt(r.shift_z, r.sig), axis=1)
    mf_tbl = mf.pivot(index='Entry', columns='property', values='cell').reindex(columns=PROPERTIES)
    mf_tbl.to_csv(shift_significance_tables_OUT / f'shift_table_model_{example_model}.csv')
    summ = m.groupby(['model', 'property'])['sig'].agg(['sum', 'count']).reset_index().rename(columns={'sum': 'n_sig', 'count': 'n_total'})
    summ['frac_sig'] = (summ.n_sig / summ.n_total).round(3)
    summ_wide = summ.pivot(index='property', columns='model', values='n_sig').reindex(PROPERTIES)
    summ_wide.to_csv(shift_significance_tables_OUT / 'shift_significance_summary.csv')
    print(f'master: {len(m)} cells ({m.Entry.nunique()} proteins x {m.model.nunique()} models x {m.property.nunique()} properties)')
    print(f'significant (FDR<0.05): {int(m.sig.sum())} / {len(m)} ({100 * m.sig.mean():.0f}%)')
    print('\n# significant cells per model (out of 25 proteins x 16 properties = 400):')
    print(m.groupby('model')['sig'].sum().sort_values(ascending=False).to_string())
    print('\n# significant cells per property (out of 25 x 7 = 175):')
    print(m.groupby('property')['sig'].sum().sort_values(ascending=False).to_string())
    print(f'\nWrote shift_significance_master.csv, shift_table_protein_{example_protein}.csv, shift_table_model_{example_model}.csv, shift_significance_summary.csv')
def shift_significance_tables__entry():
    sys.path.insert(0, str(shift_significance_tables_HERE))
    sys.path.insert(0, str(shift_significance_tables_REPO / 'design'))
    shift_significance_tables_main()

_STEPS = {
    'predict-tm-full': predict_tm_full__entry,
    'tm-shift-analysis': tm_shift_analysis__entry,
    'shift-significance-tables': shift_significance_tables__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

