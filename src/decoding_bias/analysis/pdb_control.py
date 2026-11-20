"""Rank-aware experimental-PDB control fits and matched sampling (Figure S3; Tables S7-S8).

The supplied scores are not modified.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from ..catalog import BIOPHYS_14, DOMAINS, FULL_COHORT, SCORING_CONTEXT
from .variance import make_Q, r2_from_Q, ssr_from_Q, species_effect_collapse

# The twelve models compared in Figure S3. SolubleCaliby's PDB export duplicates
# Caliby's, so it is kept only where the original complete-case rule needs it.
_CONTEXT = {'Backbone-conditioned': 'backbone', 'Structure + native sequence': 'structure+sequence',
            'Sequence-only': 'sequence'}
_ORDER = ['ProteinMPNN', 'SolubleMPNN', 'Caliby', 'ESM-IF', 'ESM3-struct', 'MIF', 'MIF-ST',
          'ESM3-seq', 'ESM2-15B', 'CARP-640M', 'ProGen2-XL', 'ProtGPT2']
PANEL = {label: (FULL_COHORT[label][0], _CONTEXT[SCORING_CONTEXT[FULL_COHORT[label][0]]])
         for label in _ORDER}
PANEL_WITH_DUPLICATE = dict(PANEL, SolubleCaliby=('soluble_caliby_score', 'unverified duplicate'))
DOMAIN_ORDER = ['Eukaryota','Bacteria','Archaea']


def matched_sample(pool, domain_counts, seed):
    """Sample without replacement, requiring every requested domain count."""
    if pool.Entry.duplicated().any():
        raise ValueError('Sampling pool contains duplicate protein identifiers')
    parts=[]
    for domain,count in domain_counts.items():
        part=pool.loc[pool.domain.eq(domain)]
        if len(part)<count:
            raise ValueError(f'{domain}: need {count} proteins but only {len(part)} available')
        parts.append(part.sample(n=int(count),random_state=int(seed)))
    return pd.concat(parts).reset_index(drop=True)


def complete_frame(frame, score_columns):
    numeric=BIOPHYS_14+list(score_columns)
    finite=np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[finite & frame.domain.isin(DOMAINS)].dropna(
        subset=['Entry','protein_family','species']).copy()


def build_bases(frame, basis_function=make_Q):
    """Fit predictor blocks on this exact cohort; ranks include the intercept."""
    std=frame[BIOPHYS_14].std(ddof=1).replace(0,1)
    z=((frame[BIOPHYS_14]-frame[BIOPHYS_14].mean())/std).to_numpy(dtype=float)
    family=pd.get_dummies(frame.protein_family,drop_first=True).to_numpy(dtype=float)
    species=pd.get_dummies(frame.species,drop_first=True).to_numpy(dtype=float)
    matrices={'Biophys':z,'Family':family,'Species':species,
              'Fam+Spec':np.column_stack([family,species]),
              'Biophys+Family':np.column_stack([z,family]),
              'Full':np.column_stack([z,family,species])}
    bases={name:basis_function(x) for name,x in matrices.items()}
    diagnostics=[{'block':name,'n':len(frame),'nominal_columns':x.shape[1]+1,
                  'rank':bases[name][1]+1,
                  'rank_deficiency':x.shape[1]-bases[name][1],
                  'residual_df':len(frame)-bases[name][1]-1}
                 for name,x in matrices.items()]
    return bases,diagnostics


def fit_table(frame, panel=PANEL, *, common=False, basis_function=make_Q):
    """Per-model or common-case fits, with explicit ranks and protein membership."""
    if frame.Entry.duplicated().any():
        raise ValueError('Analysis cohort contains duplicate protein identifiers')
    common_frame=complete_frame(frame,[c for c,_ in panel.values()]) if common else None
    rows=[];rank_rows=[];members=[];cache={}
    for model,(column,context) in panel.items():
        sub=common_frame if common else complete_frame(frame,[column])
        if len(sub)<3:
            raise ValueError(f'{model}: fewer than three complete observations')
        key=tuple(sub.Entry)
        if key not in cache:
            cache[key]=build_bases(sub,basis_function)
        bases,diagnostics=cache[key]
        y=sub[column].to_numpy(dtype=float)
        if y.std()==0:
            raise ValueError(f'{model}: constant response')
        y=(y-y.mean())/y.std()
        r={name:r2_from_Q(y,q,p) for name,(q,p) in bases.items()}
        reduced='Biophys+Family';full='Full'
        ssr_reduced=ssr_from_Q(y,bases[reduced][0])
        ssr_full=ssr_from_Q(y,bases[full][0])
        dfn=bases[full][1]-bases[reduced][1]
        dfd=len(y)-bases[full][1]-1
        f=((ssr_reduced-ssr_full)/dfn)/(ssr_full/dfd) if dfn>0 and dfd>0 else np.nan
        codes,levels=pd.factorize(sub.species)
        record={'model':model,'score_column':column,'context':context,'n':len(sub),
                'families':sub.protein_family.nunique(),'species':sub.species.nunique(),
                'n_archaea':int(sub.domain.eq('Archaea').sum()),
                'n_bacteria':int(sub.domain.eq('Bacteria').sum()),
                'n_eukaryota':int(sub.domain.eq('Eukaryota').sum()),
                'residual_species_raw':r[full][0]-r[reduced][0],
                'residual_species_adj':r[full][1]-r[reduced][1],
                'species_partial_R2':(ssr_reduced-ssr_full)/ssr_reduced,
                'species_F':f,'species_df_num':dfn,'species_df_den':dfd,
                'species_p':float(f_distribution.sf(f,dfn,dfd)) if np.isfinite(f) else np.nan}
        for name in bases:
            record[f'R2_{name}']=r[name][0]
            record[f'R2adj_{name}']=r[name][1]
            record[f'rank_{name}']=bases[name][1]+1
        record.update(species_effect_collapse(y,codes,len(levels),bases))
        rows.append(record)
        rank_rows.extend(dict(model=model,**item) for item in diagnostics)
        members.extend({'model':model,'Entry':entry} for entry in sub.Entry)
    return pd.DataFrame(rows),pd.DataFrame(rank_rows),pd.DataFrame(members)
