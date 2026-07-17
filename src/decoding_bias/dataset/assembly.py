"""decoding_bias.dataset.assembly -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - combine_main_r2_r3
  - merge_main_and_expansion
  - build_filterC_cohort
  - collapse_species_subspecies
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------- from combine_main_r2_r3.py ----------
combine_main_r2_r3_HERE = Path(__file__).parent
combine_main_r2_r3_MAIN_PATH = combine_main_r2_r3_HERE / 'Decoding_Bias_Dataset_updated.csv'
R2_PATH = combine_main_r2_r3_HERE / 'round2' / 'expansion_round2_KEPT.csv'
R3_PATH = combine_main_r2_r3_HERE / 'round3' / 'expansion_round3_for_scoring.csv'
SCORED_R2_PATH = combine_main_r2_r3_HERE / 'main_plus_round2_scored.csv'
combine_main_r2_r3_OUT_PATH = combine_main_r2_r3_HERE / 'main_plus_r2_r3.csv'
OUT_SCORED_PATH = combine_main_r2_r3_HERE / 'main_plus_r2_r3_scored.csv'
def _align_to_main(df, main_cols, source_label):
    df = df.copy()
    df['source'] = source_label
    df['structure_source'] = 'AF'
    if 'has_pdb_struct' not in df.columns:
        df['has_pdb_struct'] = False
    df['has_pdb_struct'] = df['has_pdb_struct'].fillna(False).astype(bool)
    if 'charge_at_pH7' in df.columns and 'charge_at_ph7' not in df.columns:
        df['charge_at_ph7'] = df['charge_at_pH7']
    if 'genus' not in df.columns:
        df['genus'] = np.nan
    for col in ['phylum_division', 'class', 'Description', 'has_pdb', 'alkaliphile_score', 'WT_Tm']:
        if col not in df.columns:
            df[col] = np.nan
    return df
def combine_main_r2_r3_main():
    main = pd.read_csv(combine_main_r2_r3_MAIN_PATH, low_memory=False)
    r2 = pd.read_csv(R2_PATH, low_memory=False)
    r3 = pd.read_csv(R3_PATH, low_memory=False)
    print(f'main:    {len(main)} rows')
    print(f'round-2 KEPT: {len(r2)} rows')
    print(f'round-3:      {len(r3)} rows')
    main = main.copy()
    main['source'] = 'main'
    main['structure_source'] = 'AF'
    main['has_pdb_struct'] = main['has_pdb'].fillna(False).astype(bool)
    r2 = _align_to_main(r2, main.columns, 'expansion_round2')
    r3 = _align_to_main(r3, main.columns, 'expansion_round3')
    all_cols = list(main.columns)
    for df_ in (r2, r3):
        for c in df_.columns:
            if c not in all_cols:
                all_cols.append(c)
    combined = pd.concat([main.reindex(columns=all_cols), r2.reindex(columns=all_cols), r3.reindex(columns=all_cols)], ignore_index=True)
    dups = combined['Entry'].duplicated().sum()
    if dups:
        print(f'WARNING: {dups} duplicate Entries (deduping, keeping first)')
        combined = combined.drop_duplicates('Entry', keep='first')
    combined.to_csv(combine_main_r2_r3_OUT_PATH, index=False)
    print(f'\nWrote {combine_main_r2_r3_OUT_PATH}: {len(combined)} rows x {len(combined.columns)} cols')
    print(f"  source counts: {combined['source'].value_counts().to_dict()}")
    print(f"  unique species: {combined['species'].nunique()}")
    print(f"  unique protein_family: {combined['protein_family'].nunique()}")
    print(f"  ribosomal share: {100 * (combined['broad_function'] == 'ribosomal').mean():.1f}%")
    print(f'  by domain:')
    for (d, n) in combined['domain'].value_counts().items():
        print(f'    {d:12s} {n:5d}  {100 * n / len(combined):5.1f}%')
    if SCORED_R2_PATH.exists():
        scored = pd.read_csv(SCORED_R2_PATH, low_memory=False)
        r2_scored = scored[scored['source'] == 'expansion_round2'].copy()
        score_cols = ['proteinmpnn_score', 'esmif_score', 'mif_score', 'mifst_score']
        r2_score_map = r2_scored.set_index('Entry')[score_cols]
        for col in score_cols:
            mask = combined['Entry'].isin(r2_score_map.index)
            combined.loc[mask, col] = combined.loc[mask, 'Entry'].map(r2_score_map[col]).values
        combined.to_csv(OUT_SCORED_PATH, index=False)
        print(f'\nWrote {OUT_SCORED_PATH}')
        print(f'  Round-2 KEPT score coverage:')
        r2r = combined[combined['source'] == 'expansion_round2']
        for c in score_cols:
            n = r2r[c].notna().sum()
            print(f'    {c:25s} {n}/{len(r2r)}')
        print(f'  Round-3 score coverage (all should be 0):')
        r3r = combined[combined['source'] == 'expansion_round3']
        for c in score_cols + ['ESM2_15B_pppl_score', 'carp_640M_score']:
            n = r3r[c].notna().sum()
            print(f'    {c:25s} {n}/{len(r3r)}')
def combine_main_r2_r3__entry():
    combine_main_r2_r3_main()

# ---------- from merge_main_and_expansion.py ----------
merge_main_and_expansion_HERE = Path(__file__).parent
merge_main_and_expansion_MAIN_PATH = merge_main_and_expansion_HERE / 'Decoding_Bias_Dataset_updated.csv'
EXP_PATH = merge_main_and_expansion_HERE / 'expansion_for_scoring_monomer_FINAL.csv'
AF_PATH = merge_main_and_expansion_HERE / 'scoring_results/proteinMPNN_results_all_chunks-AF_annotated.csv'
PDB_PATH = merge_main_and_expansion_HERE / 'scoring_results/proteinMPNN_results_all_chunks-PDB_annotated.csv'
ESMIF_AF_PATH = merge_main_and_expansion_HERE / 'scoring_results/esmif_AF_scores_annotated.csv'
merge_main_and_expansion_OUT_PATH = merge_main_and_expansion_HERE / 'merged_dataset.csv'
LOG_PATH = merge_main_and_expansion_HERE / 'merged_dataset_changes_log.txt'
def normalize_species(name):
    """Strip strain/parenthetical annotations: 'Bos taurus (Bovine)' -> 'Bos taurus'."""
    if pd.isna(name):
        return name
    name = re.sub('\\s*\\(strain[^)]*\\)', '', name)
    name = re.sub('\\s*\\([^)]+\\)', '', name).strip()
    return name
SEQUENCE_FEATURES = ['sequence_length', 'mw_per_residue', 'isoelectric_point', 'charge_at_ph7', 'gravy', 'instability_index', 'aromaticity', 'basic_residue_fraction', 'acidic_residue_fraction', 'ionizable_residue_fraction', 'proline_fraction', 'small_residue_fraction', 'hydrophobic_fraction', 'buffer_capacity', 'charge_per_residue']
def compute_sequence_features(seq):
    """Match src/features/sequence_features.py + annotate_scoring_results.py."""
    if not isinstance(seq, str) or not seq:
        return {k: np.nan for k in SEQUENCE_FEATURES}
    seq = seq.upper().replace('U', 'C').replace('X', '')
    if not seq:
        return {k: np.nan for k in SEQUENCE_FEATURES}
    try:
        n = len(seq)
        a = ProteinAnalysis(seq)
        aa_pct = a.get_amino_acids_percent()
        acidic = ['D', 'E']
        basic = ['K', 'R', 'H']
        ionizable = ['D', 'E', 'K', 'R', 'H', 'C', 'Y']
        small = ['A', 'G', 'S', 'T']
        hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
        charge_at_ph7 = a.charge_at_pH(7.0)
        feats = {'sequence_length': n, 'mw_per_residue': a.molecular_weight() / n, 'isoelectric_point': a.isoelectric_point(), 'charge_at_ph7': charge_at_ph7, 'gravy': a.gravy(), 'instability_index': a.instability_index(), 'aromaticity': a.aromaticity(), 'basic_residue_fraction': sum((aa_pct.get(x, 0) for x in basic)), 'acidic_residue_fraction': sum((aa_pct.get(x, 0) for x in acidic)), 'ionizable_residue_fraction': sum((aa_pct.get(x, 0) for x in ionizable)), 'proline_fraction': aa_pct.get('P', 0), 'small_residue_fraction': sum((aa_pct.get(x, 0) for x in small)), 'hydrophobic_fraction': sum((aa_pct.get(x, 0) for x in hydrophobic)), 'charge_per_residue': charge_at_ph7 / n, 'buffer_capacity': abs(a.charge_at_pH(8.0) - a.charge_at_pH(6.0)) / 2.0}
        return feats
    except Exception:
        return {k: np.nan for k in SEQUENCE_FEATURES}
OUTPUT_COLS = ['Entry', 'source', 'structure_source', 'has_pdb_struct', 'sequence', 'sequence_length', 'species', 'species_raw', 'domain', 'phylum_division', 'class', 'genus', 'protein_name', 'protein_name_clean', 'protein_family', 'broad_function', 'Description', 'mw_per_residue', 'isoelectric_point', 'charge_at_ph7', 'gravy', 'instability_index', 'aromaticity', 'basic_residue_fraction', 'acidic_residue_fraction', 'ionizable_residue_fraction', 'proline_fraction', 'small_residue_fraction', 'hydrophobic_fraction', 'buffer_capacity', 'charge_per_residue', 'alkaliphile_score', 'avg_plddt', 'helix_percent', 'sheet_percent', 'rco', 'surface_exposure', 'avg_cb_distance', 'compactness', 'is_enzyme', 'is_transmembrane', 'is_glycosylated', 'has_disordered', 'has_pdb', 'WT_Tm', 'proteinmpnn_score', 'esmif_score', 'mif_score', 'mifst_score', 'carp_640M_score', 'ESM2_15B_pppl_score', 'AlkSecMPNN_score', 'caliby_score', 'triflow_score', 'esm3_struct_cond_score', 'esm3_seq_only_score']
def merge_main_and_expansion_main():
    log = []
    log.append('=' * 70)
    log.append('MERGE LOG')
    log.append('=' * 70)
    main_df = pd.read_csv(merge_main_and_expansion_MAIN_PATH)
    log.append(f'Loaded main: {len(main_df)} proteins')
    main_df['source'] = 'main'
    main_df['structure_source'] = 'AF'
    main_df['has_pdb_struct'] = main_df['has_pdb'].astype(bool)
    main_df['species_raw'] = main_df['species']
    main_df['species'] = main_df['species'].apply(normalize_species)
    exp_curated = pd.read_csv(EXP_PATH)
    af_scored = pd.read_csv(AF_PATH)
    pdb_scored = pd.read_csv(PDB_PATH)
    log.append(f'Loaded expansion: curated={len(exp_curated)}, AF-scored={len(af_scored)}, PDB-scored={len(pdb_scored)}')
    af_entries = set(af_scored['Entry'])
    pdb_entries = set(pdb_scored['Entry'])
    viable_entries = af_entries | pdb_entries
    unusable = set(exp_curated['Entry']) - viable_entries
    log.append(f'Expansion proteins with NO usable structure (dropped): {len(unusable)}')
    log.append(f'Expansion viable (AF-only / PDB-only / both): {len(af_entries - pdb_entries)} / {len(pdb_entries - af_entries)} / {len(af_entries & pdb_entries)} = {len(viable_entries)} total')
    af_scored = af_scored.copy()
    af_scored['structure_source'] = 'AF'
    pdb_only = pdb_scored[~pdb_scored['Entry'].isin(af_entries)].copy()
    pdb_only['structure_source'] = 'PDB'
    pdb_only['avg_plddt'] = np.nan
    exp_rows = pd.concat([af_scored, pdb_only], ignore_index=True)
    log.append(f'Expansion rows in merge: {len(exp_rows)}')
    exp_rows['species_raw'] = exp_rows['species']
    exp_rows['species'] = exp_rows['species'].apply(normalize_species)
    log.append(f"Unique normalized species: main={main_df['species'].nunique()}, expansion={exp_rows['species'].nunique()}, shared={len(set(main_df['species']) & set(exp_rows['species']))}")
    exp_rows['source'] = 'expansion_' + exp_rows['structure_source']
    exp_rows['has_pdb_struct'] = exp_rows['Entry'].isin(pdb_entries)
    if ESMIF_AF_PATH.exists():
        esmif_af = pd.read_csv(ESMIF_AF_PATH)
        score_col = next((c for c in esmif_af.columns if 'score' in c.lower() and 'valid' not in c and ('pos_' not in c) and ('missing' not in c) and ('std' not in c) and ('min' not in c) and ('max' not in c)), None)
        if score_col is None:
            score_col = 'esmif_af_score'
        if score_col in esmif_af.columns:
            esmif_map = esmif_af.set_index('Entry')[score_col].to_dict()
            exp_rows['esmif_score'] = exp_rows['Entry'].map(esmif_map)
            log.append(f"Filled esmif_score from {ESMIF_AF_PATH.name} (column {score_col}): {exp_rows['esmif_score'].notna().sum()} proteins")
        else:
            log.append('ESM-IF AF score column not found; esmif_score left NaN for expansion')
    if 'sequence_score' in exp_rows.columns:
        exp_rows['proteinmpnn_score'] = exp_rows['sequence_score']
    log.append('Computing BioPython sequence features for expansion rows...')
    feat_records = []
    for (_, row) in exp_rows.iterrows():
        feats = compute_sequence_features(str(row['sequence']))
        feats['Entry'] = row['Entry']
        feat_records.append(feats)
    feat_df = pd.DataFrame(feat_records)
    overlap = (set(feat_df.columns) & set(exp_rows.columns)) - {'Entry'}
    exp_rows = exp_rows.drop(columns=list(overlap))
    exp_rows = exp_rows.merge(feat_df, on='Entry', how='left')
    log.append(f'Computed {len(feat_df.columns) - 1} biophysical features for {len(feat_df)} proteins')
    main_bf = set(main_df['broad_function'].unique())
    exp_bf = set(exp_rows['broad_function'].unique())
    only_main = main_bf - exp_bf
    only_exp = exp_bf - main_bf
    log.append(f'broad_function only in main: {sorted(only_main)}')
    log.append(f'broad_function only in expansion: {sorted(only_exp)}')
    log.append('(Vocabulary unified as union; no remapping performed)')
    for c in OUTPUT_COLS:
        if c not in main_df.columns:
            main_df[c] = np.nan
        if c not in exp_rows.columns:
            exp_rows[c] = np.nan
    main_keep = main_df[OUTPUT_COLS]
    exp_keep = exp_rows[OUTPUT_COLS]
    merged = pd.concat([main_keep, exp_keep], ignore_index=True)
    log.append(f'\nFinal merged dataset: {len(merged)} proteins')
    log.append(f"  by source: {merged['source'].value_counts().to_dict()}")
    log.append(f"  by structure_source: {merged['structure_source'].value_counts().to_dict()}")
    log.append(f"  with PDB structure: {int(merged['has_pdb_struct'].sum())}")
    log.append(f"  unique species: {merged['species'].nunique()}")
    log.append(f"  unique protein_family: {merged['protein_family'].nunique()}")
    log.append(f"  unique broad_function: {merged['broad_function'].nunique()}")
    log.append(f"  ribosomal share: {100 * (merged['broad_function'] == 'ribosomal').mean():.1f}%")
    merged.to_csv(merge_main_and_expansion_OUT_PATH, index=False)
    LOG_PATH.write_text('\n'.join(log) + '\n')
    print('\n'.join(log))
    print(f'\nWrote {merge_main_and_expansion_OUT_PATH}')
    print(f'Wrote {LOG_PATH}')
def merge_main_and_expansion__entry():
    merge_main_and_expansion_main()

# ---------- from build_filterC_cohort.py ----------
build_filterC_cohort_HERE = Path(__file__).parent
RAW = build_filterC_cohort_HERE / 'main_plus_r2_r3_speciescollapsed.csv'
SCORED = build_filterC_cohort_HERE / 'main_plus_r2_r3_scored.csv'
OUT = build_filterC_cohort_HERE / 'main_plus_r2_r3_filterC.csv'
OUT_SCORED = build_filterC_cohort_HERE / 'main_plus_r2_r3_scored_filterC.csv'
SP_COL = 'species_collapsed'
MIN_SPECIES_PER_FAMILY = 5
MIN_PROTEINS_PER_SPECIES = 2
def apply_filter_C(d):
    """Iteratively enforce filter C until stable."""
    while True:
        n0 = len(d)
        fam_sp = d.groupby('protein_family')[SP_COL].nunique()
        keep_fam = set(fam_sp[fam_sp >= MIN_SPECIES_PER_FAMILY].index)
        d = d[d['protein_family'].isin(keep_fam)]
        sp_count = d[SP_COL].value_counts()
        keep_sp = set(sp_count[sp_count >= MIN_PROTEINS_PER_SPECIES].index)
        d = d[d[SP_COL].isin(keep_sp)]
        if len(d) == n0:
            break
    return d
def build_filterC_cohort_main():
    df = pd.read_csv(RAW, low_memory=False)
    print(f'Loaded {len(df)} proteins from {RAW.name}')
    filtered = apply_filter_C(df)
    filtered.to_csv(OUT, index=False)
    print(f'Wrote {OUT.name}: {len(filtered)} proteins (removed {len(df) - len(filtered)} = {100 * (len(df) - len(filtered)) / len(df):.1f}%)')
    print(f'  unique species (collapsed): {filtered[SP_COL].nunique()}')
    print(f"  unique protein_family:      {filtered['protein_family'].nunique()}")
    fam_dom = filtered.groupby('protein_family')['domain'].nunique()
    print(f'  multi-domain families:      {(fam_dom >= 2).sum()} / {len(fam_dom)}')
    print(f'  3-domain families:          {(fam_dom == 3).sum()}')
    if SCORED.exists():
        scored = pd.read_csv(SCORED, low_memory=False)
        sp_map = filtered.set_index('Entry')[SP_COL]
        scored_keep = scored[scored['Entry'].isin(set(filtered['Entry']))].copy()
        scored_keep[SP_COL] = scored_keep['Entry'].map(sp_map)
        scored_keep.to_csv(OUT_SCORED, index=False)
        print(f'\nWrote {OUT_SCORED.name}: {len(scored_keep)} proteins (scored cohort)')
def build_filterC_cohort__entry():
    build_filterC_cohort_main()

# ---------- from collapse_species_subspecies.py ----------
collapse_species_subspecies_HERE = Path(__file__).parent
IN_CSV = collapse_species_subspecies_HERE / 'main_plus_r2_r3.csv'
OUT_CSV = collapse_species_subspecies_HERE / 'main_plus_r2_r3_speciescollapsed.csv'
SUBSPECIES_TAGS = re.compile('\\s+(subsp\\.|serotype|serovar|biotype|biovar)\\s+.*$', flags=re.IGNORECASE)
PATHOTYPE_TAIL = re.compile('\\s+[OKH]\\d+(?::[OKH]\\d+)*(?:\\s*[\\w/]*)?$', flags=re.IGNORECASE)
def collapse(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    if re.search('\\bsp\\.\\s*$', s):
        return s
    s = SUBSPECIES_TAGS.sub('', s)
    s = PATHOTYPE_TAIL.sub('', s)
    return s.strip()
def collapse_species_subspecies_main():
    df = pd.read_csv(IN_CSV, low_memory=False)
    df['species_collapsed'] = df['species'].apply(collapse)
    before = df['species'].nunique()
    after = df['species_collapsed'].nunique()
    df.to_csv(OUT_CSV, index=False)
    print(f'main_plus_r2_r3 species labels:')
    print(f'  before collapse: {before}')
    print(f'  after collapse:  {after}')
    print(f'  collapsed away:  {before - after}')
    affected = df[df['species'] != df['species_collapsed']]
    if len(affected):
        merges = affected.groupby('species_collapsed')['species'].nunique().sort_values(ascending=False).head(10)
        print(f'\nTop merges (target species ← number of variants pooled in):')
        for (sp_c, n) in merges.items():
            variants = affected[affected['species_collapsed'] == sp_c]['species'].unique()
            tot = (df['species_collapsed'] == sp_c).sum()
            print(f'  {sp_c:50s} ({n} variants → {tot} proteins)')
    print(f'\nWrote {OUT_CSV}')
def collapse_species_subspecies__entry():
    collapse_species_subspecies_main()

_STEPS = {
    'combine-main-r2-r3': combine_main_r2_r3__entry,
    'merge-main-and-expansion': merge_main_and_expansion__entry,
    'build-filterC-cohort': build_filterC_cohort__entry,
    'collapse-species-subspecies': collapse_species_subspecies__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

