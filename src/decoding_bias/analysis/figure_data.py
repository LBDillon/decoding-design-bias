"""Data behind the unweighted Figure 2 panels.

The 2,000-draw species bootstrap conditions on the fitted species ratings; it does
not rerun Elo matches or resample protein families.
"""
import numpy as np
import pandas as pd

from .. import catalog

# Bootstrap draws share one generator, so this order also fixes the intervals.
MODEL_ORDER = catalog.ELO_FULL_COLUMNS
LABELS = catalog.LABELS
CLASSES = catalog.SCORING_CONTEXT
ALIASES = {'Firmicutes':'Bacillota','Proteobacteria':'Pseudomonadota','Cyanobacteria':'Cyanobacteriota',
           'Actinobacteria':'Actinomycetota','Bacteroidetes':'Bacteroidota'}


def elo_panels(ratings, taxonomy, n_boot=2000, seed=0):
    """Return bootstrap gaps, displayed species, phylum z means, metadata, species z.

    Each species has equal weight. Phylum cells average species-level z-scores,
    where the z-score denominator is the population SD across all species in a
    model. Models retain the paper's scoring-context order. Rows are grouped by
    domain then descending across-model mean, as in the original figure code.
    """
    ratings = ratings.copy()
    assert not ratings.duplicated(['species','model']).any()
    models = [m for m in MODEL_ORDER if m in set(ratings.model)]
    rng = np.random.default_rng(seed)
    rows = []
    for m in models:
        sub = ratings[ratings.model.eq(m)]
        a = sub.loc[sub.domain.eq('Archaea'),'rating'].to_numpy()
        e = sub.loc[sub.domain.eq('Eukaryota'),'rating'].to_numpy()
        gap = a.mean()-e.mean()
        samples = np.array([rng.choice(a,a.size,True).mean()-rng.choice(e,e.size,True).mean() for _ in range(n_boot)])
        lo,hi = np.percentile(samples,[2.5,97.5])
        rows.append({'model':m,'label':LABELS[m],'gap':gap,'ci_lower':lo,'ci_upper':hi,
                     'n_archaea':len(a),'n_eukaryota':len(e),'bootstrap_replicates':n_boot,'seed':seed,'scoring_context':CLASSES[m]})
    gaps = pd.DataFrame(rows)
    species = ratings[ratings.model.isin(['proteinmpnn_score','protgpt2_score'])].copy()
    species['domain_order'] = species.domain.map({'Archaea':0,'Bacteria':1,'Eukaryota':2})
    species = species.sort_values(['model','domain_order','rating','species']).reset_index(drop=True)
    species['plot_position'] = species.groupby('model').cumcount()
    species['rating_minus_1500'] = species.rating-1500
    tax = taxonomy[['species','phylum_division']].drop_duplicates()
    assert not tax.species.duplicated().any()
    z = ratings.merge(tax,on='species',how='left',validate='many_to_one')
    z['phylum'] = z.phylum_division.fillna('Other').replace(ALIASES)
    z['species_z'] = z.groupby('model').rating.transform(lambda x:(x-x.mean())/(x.std(ddof=0) or 1.))
    matrix = z.groupby(['phylum','model']).species_z.mean().unstack().reindex(columns=models)
    domains = z.groupby('phylum').domain.agg(lambda x:x.mode().iat[0])
    order=[]
    for dom in ['Eukaryota','Archaea','Bacteria']:
        order += sorted([p for p in matrix.index if domains[p]==dom],key=lambda p:matrix.loc[p].mean(),reverse=True)
    matrix=matrix.reindex(order)
    counts=z.drop_duplicates(['species','phylum']).groupby('phylum').species.nunique()
    meta=pd.DataFrame({'phylum':order,'domain':[domains[p] for p in order],
                       'n_species':[counts[p] for p in order],'row_order':range(len(order))})
    return gaps,species,matrix,meta,z
