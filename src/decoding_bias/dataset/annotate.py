"""decoding_bias.dataset.annotate -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - annotate_scoring_results
  - build_v12_features_consistent
  - make_corrected_v12_csv
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import requests
import sys
import time
import warnings
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm
from features_for_designs import sequence_features, MIXED_FEATURES
from decoding_bias.features.sequence_features import calculate_sequence_features

# ---------- from annotate_scoring_results.py ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXISTING = REPO_ROOT / 'dataset_update' / 'Decoding_Bias_Dataset_updated.csv'
EXP_AF = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_FINAL.csv'
EXP_PDB = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_PDB_FINAL.csv'
NEEDED_COLS = ['species', 'domain', 'sequence', 'protein_name', 'protein_family', 'broad_function', 'sequence_length', 'isoelectric_point', 'charge_at_ph7', 'gravy', 'instability_index', 'mw_per_residue', 'aromaticity', 'helix_percent', 'sheet_percent', 'surface_exposure', 'avg_cb_distance', 'rco', 'avg_plddt', 'compactness']
SEQ_FEATURES = ['isoelectric_point', 'charge_at_ph7', 'gravy', 'instability_index', 'mw_per_residue', 'aromaticity']
STRUCTURAL_FEATURES = ['helix_percent', 'sheet_percent', 'surface_exposure', 'avg_cb_distance', 'rco', 'compactness']
def compute_seq_features(seq: str) -> dict:
    """Sequence-derived features for one protein. Mirrors 02_calculate_sequence_features.py."""
    if not isinstance(seq, str) or not seq:
        return {f: None for f in SEQ_FEATURES + ['sequence_length']}
    seq = seq.upper().replace('U', 'C').replace('X', '')
    if not seq:
        return {f: None for f in SEQ_FEATURES + ['sequence_length']}
    try:
        a = ProteinAnalysis(seq)
        L = len(seq)
        return {'sequence_length': L, 'mw_per_residue': a.molecular_weight() / L, 'aromaticity': a.aromaticity(), 'instability_index': a.instability_index(), 'isoelectric_point': a.isoelectric_point(), 'charge_at_ph7': a.charge_at_pH(7.0), 'gravy': a.gravy()}
    except Exception:
        return {f: None for f in SEQ_FEATURES + ['sequence_length']}
def fetch_uniprot_taxa(entries: list[str]) -> dict[str, dict]:
    """Query UniProt for species + taxonomic lineage for entries missing those.

    Returns dict mapping Entry → {species, domain, organism_full}.
    Uses the public REST endpoint (rate-limited; one request per entry).
    """
    out = {}
    for entry in entries:
        url = f'https://rest.uniprot.org/uniprotkb/{entry}.json'
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                out[entry] = {}
                continue
            d = r.json()
            org = d.get('organism', {})
            sci = org.get('scientificName', '')
            lineage = org.get('lineage', [])
            domain = next((t for t in ('Viruses', 'Archaea', 'Bacteria', 'Eukaryota') if t in lineage), 'Unknown')
            short = ' '.join(sci.split()[:2]) if sci else ''
            out[entry] = {'species': short, 'domain': domain, 'organism_full': sci}
        except requests.RequestException:
            out[entry] = {}
        time.sleep(0.05)
    return out
def annotate(scores_path: Path, out_dir: Path, label: str='') -> None:
    print(f'\n=== Annotating {scores_path.name} ===')
    scores = pd.read_csv(scores_path)
    print(f'  loaded {len(scores)} rows')
    print('Loading reference datasets...')
    existing = pd.read_csv(EXISTING, low_memory=False)
    exp_af = pd.read_csv(EXP_AF, low_memory=False)
    exp_pdb = pd.read_csv(EXP_PDB, low_memory=False) if EXP_PDB.exists() else pd.DataFrame()
    annotated = scores.copy()
    annotated['source'] = None
    existing_keep = ['Entry'] + [c for c in NEEDED_COLS if c in existing.columns]
    ann_existing = existing[existing_keep].rename(columns={})
    annotated = annotated.merge(ann_existing, on='Entry', how='left', suffixes=('', '_ex'))
    matched_existing = annotated[NEEDED_COLS[0]].notna()
    annotated.loc[matched_existing, 'source'] = 'existing_dataset'
    print(f'  matched in existing dataset: {matched_existing.sum()}')
    unmatched_mask = annotated['source'].isna()
    if unmatched_mask.any():
        print(f'  unmatched after existing: {unmatched_mask.sum()}')
        for (exp_df, label_src) in [(exp_af, 'expansion_AF'), (exp_pdb, 'expansion_PDB')]:
            if exp_df.empty:
                continue
            exp_keep = ['Entry']
            for c in ['Organism', 'domain', 'sequence', 'protein_name_clean', 'protein_family', 'broad_function']:
                if c in exp_df.columns:
                    exp_keep.append(c)
            sub = exp_df[exp_keep].copy()
            sub = sub.rename(columns={'Organism': 'species_exp', 'protein_name_clean': 'protein_name_exp', 'sequence': 'sequence_exp', 'domain': 'domain_exp', 'protein_family': 'protein_family_exp', 'broad_function': 'broad_function_exp'})
            annotated = annotated.merge(sub, on='Entry', how='left')
            for (col_orig, col_new) in [('species', 'species_exp'), ('domain', 'domain_exp'), ('sequence', 'sequence_exp'), ('protein_name', 'protein_name_exp'), ('protein_family', 'protein_family_exp'), ('broad_function', 'broad_function_exp')]:
                if col_new in annotated.columns:
                    fill = annotated[col_orig].isna() & annotated[col_new].notna()
                    annotated.loc[fill, col_orig] = annotated.loc[fill, col_new]
                    annotated.loc[fill, 'source'] = label_src
                    annotated.drop(columns=[col_new], inplace=True)
    needs_seq_feat = annotated[SEQ_FEATURES[0]].isna() & annotated['sequence'].notna()
    if needs_seq_feat.any():
        print(f'  computing biophysical features for {needs_seq_feat.sum()} entries...')
        for idx in annotated[needs_seq_feat].index:
            feats = compute_seq_features(annotated.at[idx, 'sequence'])
            for (k, v) in feats.items():
                annotated.at[idx, k] = v
    missing_struct = annotated[STRUCTURAL_FEATURES[0]].isna()
    annotated['needs_structural_extraction'] = missing_struct
    annotated['needs_avg_plddt'] = annotated['avg_plddt'].isna()
    print(f'  needs structural extraction: {missing_struct.sum()}')
    print(f"  needs avg_plddt:             {annotated['needs_avg_plddt'].sum()}")
    out_dir.mkdir(exist_ok=True)
    base = scores_path.stem
    out_main = out_dir / f'{base}_annotated.csv'
    annotated.to_csv(out_main, index=False)
    print(f'\nSaved annotated CSV: {out_main}')
    missing = annotated[annotated['species'].isna() | annotated['domain'].isna() | annotated['protein_family'].isna()]
    if not missing.empty:
        out_miss = out_dir / f'{base}_missing_annotations.csv'
        missing[['Entry', 'species', 'domain', 'protein_family', 'source']].to_csv(out_miss, index=False)
        print(f'  warning: {len(missing)} rows missing species/domain/protein_family - see {out_miss.name}')
        print(f'  → consider running fetch_uniprot_taxa() on these')
    full = annotated[annotated[['species', 'domain', 'protein_family', 'sequence_length', 'isoelectric_point', 'gravy']].notna().all(axis=1)]
    print(f'\n  ready for VD (with seq features only):    {len(full):,} / {len(annotated):,}')
    full_struct = full[full[STRUCTURAL_FEATURES + ['avg_plddt']].notna().all(axis=1)]
    print(f'  ready for VD (with structural + pLDDT):   {len(full_struct):,} / {len(annotated):,}')
def annotate_scoring_results_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores_csv', type=Path, nargs='+', help='ProteinMPNN result CSV(s) to annotate')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'dataset_update' / 'scoring_results')
    args = parser.parse_args()
    for p in args.scores_csv:
        annotate(p, args.output_dir)
def annotate_scoring_results__entry():
    annotate_scoring_results_main()

# ---------- from build_v12_features_consistent.py ----------
build_v12_features_consistent_REPO = Path(__file__).resolve().parent.parent
build_v12_features_consistent_ANA = build_v12_features_consistent_REPO / 'dataset_update' / 'main_plus_r2_r3_analysis_v12.csv'
build_v12_features_consistent_META = build_v12_features_consistent_REPO / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
build_v12_features_consistent_OUT = Path(__file__).resolve().parent / 'outputs' / 'v12_features_consistent.csv'
SEQ_FEATS = ['sequence_length', 'mw_per_residue', 'isoelectric_point', 'charge_at_ph7', 'acidic_residue_fraction', 'basic_residue_fraction', 'gravy', 'aromaticity', 'instability_index', 'proline_fraction', 'small_residue_fraction']
STRUCT_FEATS = ['ordered_percent', 'helix_sheet_contrast', 'rco', 'avg_cb_distance', 'surface_exposure']
SCORE_COLS = ['proteinmpnn_score', 'solublempnn_score', 'esmif_score', 'mif_score', 'mifst_score', 'ESM2_15B_pppl_score', 'caliby_score', 'triflow_score', 'esm3_struct_cond_score', 'esm3_seq_only_score', 'carp_640M_score']
def build_v12_features_consistent_clean_seq(s):
    """Same cleaning used by the v12 builder: U->C (selenocysteine), drop X."""
    if not isinstance(s, str):
        return ''
    return s.upper().replace('U', 'C').replace('X', '')
def build_v12_features_consistent_main():
    ana = pd.read_csv(build_v12_features_consistent_ANA, low_memory=False)
    meta = pd.read_csv(build_v12_features_consistent_META, low_memory=False)[['Entry', 'sequence']]
    df = ana.merge(meta, on='Entry', how='left')
    print(f'v12 cohort: {len(df)} proteins')
    rows = []
    for s in tqdm(df['sequence'], desc='sequence features'):
        cs = build_v12_features_consistent_clean_seq(s)
        if not cs:
            rows.append({k: np.nan for k in SEQ_FEATS})
            continue
        try:
            f = sequence_features(cs)
            rows.append({k: f.get(k) for k in SEQ_FEATS})
        except Exception:
            rows.append({k: np.nan for k in SEQ_FEATS})
    seqf = pd.DataFrame(rows)
    out = pd.DataFrame({'Entry': df['Entry'], 'domain': df['domain']})
    for k in SEQ_FEATS:
        out[k] = seqf[k].values
    for k in STRUCT_FEATS:
        out[k] = df[k].values
    score_cols = [c for c in df.columns if c.endswith('_score')]
    for k in score_cols:
        out[k] = df[k].values
    print(f'carried {len(score_cols)} score columns: {score_cols}')
    build_v12_features_consistent_OUT.parent.mkdir(exist_ok=True)
    out.to_csv(build_v12_features_consistent_OUT, index=False)
    print(f'\nWrote {build_v12_features_consistent_OUT}  ({out.shape[0]} rows, {out.shape[1]} cols)')
    miss = {k: int(out[k].isna().sum()) for k in MIXED_FEATURES if out[k].isna().sum()}
    print('Missing per feature:', miss or 'none')
    return out
def build_v12_features_consistent__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    build_v12_features_consistent_main()

# ---------- from make_corrected_v12_csv.py ----------
make_corrected_v12_csv_REPO = Path(__file__).resolve().parent.parent
make_corrected_v12_csv_ANA = make_corrected_v12_csv_REPO / 'dataset_update' / 'main_plus_r2_r3_analysis_v12.csv'
make_corrected_v12_csv_META = make_corrected_v12_csv_REPO / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
make_corrected_v12_csv_OUT = make_corrected_v12_csv_REPO / 'dataset_update' / 'main_plus_r2_r3_analysis_v12_corrected.csv'
COLMAP = {'sequence_length': 'sequence_length', 'mw_per_residue': 'mw_per_residue', 'isoelectric_point': 'isoelectric_point', 'charge_at_ph7': 'charge_at_pH7', 'charge_per_residue': 'charge_per_residue', 'buffer_capacity': 'buffer_capacity', 'basic_residue_fraction': 'basic_residue_fraction', 'acidic_residue_fraction': 'acidic_residue_fraction', 'gravy': 'gravy', 'aromaticity': 'aromaticity', 'hydrophobic_fraction': 'hydrophobic_fraction', 'instability_index': 'instability_index', 'proline_fraction': 'proline_fraction', 'small_residue_fraction': 'small_residue_fraction'}
def make_corrected_v12_csv_clean_seq(s):
    return s.upper().replace('U', 'C').replace('X', '') if isinstance(s, str) else ''
def make_corrected_v12_csv_main():
    ana = pd.read_csv(make_corrected_v12_csv_ANA, low_memory=False)
    seqs = pd.read_csv(make_corrected_v12_csv_META, low_memory=False).set_index('Entry')['sequence']
    ana['_seq'] = ana['Entry'].map(seqs)
    refreshed = {c: [] for c in COLMAP}
    for s in tqdm(ana['_seq'], desc='recomputing sequence features'):
        cs = make_corrected_v12_csv_clean_seq(s)
        f = calculate_sequence_features(cs) if cs else {}
        for (col, key) in COLMAP.items():
            refreshed[col].append(f.get(key, np.nan) if f else np.nan)
    print('\nColumn changes (mean relative |Δ| over rows with both values):')
    corrected = ana.copy()
    for col in COLMAP:
        if col not in ana.columns:
            print(f'  {col:26} (not in analysis_v12 - adding)')
            corrected[col] = refreshed[col]
            continue
        old = ana[col].astype(float)
        new = pd.Series(refreshed[col], index=ana.index)
        both = old.notna() & new.notna() & (old.abs() > 1e-09)
        rel = ((new[both] - old[both]).abs() / old[both].abs()).mean()
        tag = 'unchanged' if rel < 0.005 else 'CORRECTED' if rel > 0.05 else 'minor'
        print(f'  {col:26} mean rel Δ = {rel:6.1%}   {tag}')
        corrected[col] = refreshed[col]
    corrected = corrected.drop(columns=['_seq'])
    corrected.to_csv(make_corrected_v12_csv_OUT, index=False)
    print(f'\nWrote {make_corrected_v12_csv_OUT}')
    print(f"  shape {corrected.shape} (same rows/cols as original: {corrected.shape == ana.drop(columns=['_seq']).shape})")
def make_corrected_v12_csv__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(make_corrected_v12_csv_REPO))
    make_corrected_v12_csv_main()

_STEPS = {
    'annotate-scoring-results': annotate_scoring_results__entry,
    'build-v12-features-consistent': build_v12_features_consistent__entry,
    'make-corrected-v12-csv': make_corrected_v12_csv__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

