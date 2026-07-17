"""decoding_bias.pdb_cohort.cohort_scoring -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - prep_independent_cohort_structures
  - prep_pdb_inputs_fresh
  - build_pdb_cohort_features
  - score_esmif_cohort
  - merge_pdb_cohort_scores
"""

import argparse
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio
import biotite.structure.io.pdbx as pdbx
import esm
import glob
import numpy as np
import os
import pandas as pd
import sys
import torch
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from biotite.sequence import ProteinSequence
from build_pdb_scoring_inputs import three_to_one, best_alignment, MAX_CIF_MB, CIF_CACHE, CHAIN_DIR, IDENTITY_MIN, SCORE_COLS
from tqdm import tqdm
from features_for_designs import sequence_features, structure_features, MIXED_FEATURES

# ---------- from prep_independent_cohort_structures.py ----------
prep_independent_cohort_structures_HERE = Path(__file__).resolve().parent
prep_independent_cohort_structures_COH = prep_independent_cohort_structures_HERE / 'outputs' / 'independent_cohort'
IN_CSV = prep_independent_cohort_structures_COH / 'cohort_scoring_inputs.csv'
prep_independent_cohort_structures_CIF_CACHE = prep_independent_cohort_structures_COH / '_cif_cache'
prep_independent_cohort_structures_CHAIN_DIR = prep_independent_cohort_structures_COH / 'cohort_chain_structs'
prep_independent_cohort_structures_PARTIAL = prep_independent_cohort_structures_COH / 'cohort_pdb_scoring_inputs.partial.csv'
MIN_LEN = 50
prep_independent_cohort_structures_MAX_CIF_MB = 15
MODIFIED = {'MSE': 'M', 'SEP': 'S', 'TPO': 'T', 'PTR': 'Y', 'HYP': 'P', 'PCA': 'E', 'CME': 'C', 'CSO': 'C', 'KCX': 'K', 'MLY': 'K', 'LLP': 'K', 'CSD': 'C', 'OCS': 'C', 'CAS': 'C'}
STD_AA = set('ACDEFGHIKLMNPQRSTVWY')
META_COLS = ['domain', 'species_collapsed', 'protein_family', 'broad_function', 'resolution_A']
def prep_independent_cohort_structures_three_to_one(rn):
    try:
        return ProteinSequence.convert_letter_3to1(rn)
    except Exception:
        return MODIFIED.get(rn, 'X')
def prep_independent_cohort_structures_process(row):
    e = row['entity_id']
    pdb_id = str(row['pdb_id']).strip().upper()
    want_chain = str(row['chain']).strip()
    out = {'Entry': e, 'pdb_id': pdb_id, 'chain_req': want_chain, 'fail_reason': None}
    try:
        cif = prep_independent_cohort_structures_CIF_CACHE / f'{pdb_id}.cif'
        if not cif.exists():
            rcsb.fetch(pdb_id, 'cif', str(prep_independent_cohort_structures_CIF_CACHE))
        if cif.stat().st_size > prep_independent_cohort_structures_MAX_CIF_MB * 1000000.0:
            out['fail_reason'] = 'large_assembly_skipped'
            return out
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif)), model=1)
        aa = arr[struc.filter_amino_acids(arr)]
        chains = sorted(set(aa.chain_id))
        ch = want_chain if want_chain in chains else chains[0] if len(chains) == 1 else None
        if ch is None:
            out['fail_reason'] = f'chain_{want_chain}_absent'
            return out
        ca = aa[(aa.chain_id == ch) & (aa.atom_name == 'CA')]
        if ca.array_length() == 0:
            out['fail_reason'] = 'no_CA'
            return out
        seq = ''.join((prep_independent_cohort_structures_three_to_one(r) for r in ca.res_name))
        out.update(chain=ch, resolved_len=len(seq), seqres_len=len(str(row.get('sequence') or '')))
        if set(seq) - STD_AA:
            out['fail_reason'] = 'nonstandard_residue'
            return out
        if len(seq) < MIN_LEN:
            out['fail_reason'] = f'resolved_len_{len(seq)}<{MIN_LEN}'
            return out
        chain_atoms = aa[aa.chain_id == ch]
        chain_atoms.chain_id[:] = 'A'
        cp = prep_independent_cohort_structures_CHAIN_DIR / f'{e}_{pdb_id}_{ch}.pdb'
        pf = pdbio.PDBFile()
        pf.set_structure(chain_atoms)
        pf.write(str(cp))
        out['chain_pdb_path'] = str(cp)
        out['sequence'] = seq
    except Exception as ex:
        out['fail_reason'] = f'error:{str(ex)[:50]}'
    return out
def prep_independent_cohort_structures_main():
    df = pd.read_csv(IN_CSV)
    meta = df.set_index('entity_id')[META_COLS]
    done = set()
    if prep_independent_cohort_structures_PARTIAL.exists():
        done = set(pd.read_csv(prep_independent_cohort_structures_PARTIAL)['Entry'])
        print(f'resuming, {len(done)} done')
    todo = df[~df['entity_id'].isin(done)].reset_index(drop=True)
    print(f'cohort: {len(df)}  | to process: {len(todo)}')
    results = pd.read_csv(prep_independent_cohort_structures_PARTIAL).to_dict('records') if prep_independent_cohort_structures_PARTIAL.exists() else []
    rowdicts = todo.to_dict('records')
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(prep_independent_cohort_structures_process, r) for r in rowdicts]
        for (i, f) in enumerate(tqdm(as_completed(futs), total=len(futs), desc='cohort PDB prep')):
            results.append(f.result())
            if i % 50 == 0:
                pd.DataFrame(results).to_csv(prep_independent_cohort_structures_PARTIAL, index=False)
    info = pd.DataFrame(results)
    ok = info[info.fail_reason.isna() & info.get('chain_pdb_path').notna()].copy()
    ok = ok.merge(meta, left_on='Entry', right_index=True, how='left')
    keep = ['Entry', 'pdb_id', 'chain', 'sequence', 'chain_pdb_path', 'domain', 'species_collapsed', 'protein_family', 'broad_function', 'resolution_A', 'resolved_len', 'seqres_len']
    ok[keep].to_csv(prep_independent_cohort_structures_COH / 'cohort_pdb_scoring_inputs.csv', index=False)
    info.to_csv(prep_independent_cohort_structures_COH / 'cohort_prep_diagnostics.csv', index=False)
    if prep_independent_cohort_structures_PARTIAL.exists():
        prep_independent_cohort_structures_PARTIAL.unlink()
    print(f'\nUsable scoring inputs: {len(ok)} / {len(df)}')
    print('By domain:')
    print(ok['domain'].value_counts().to_string())
    print('\nReasons excluded:')
    print(info.fail_reason.value_counts(dropna=True).to_string())
    print(f'\nWrote cohort_pdb_scoring_inputs.csv (sequence = resolved chain seq; chain PDBs in {prep_independent_cohort_structures_CHAIN_DIR.name}/)')
def prep_independent_cohort_structures__entry():
    warnings.filterwarnings('ignore')
    prep_independent_cohort_structures_CIF_CACHE.mkdir(parents=True, exist_ok=True)
    prep_independent_cohort_structures_CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    prep_independent_cohort_structures_main()

# ---------- from prep_pdb_inputs_fresh.py ----------
prep_pdb_inputs_fresh_HERE = Path(__file__).resolve().parent
prep_pdb_inputs_fresh_REPO = Path(__file__).resolve().parent.parent
AVAIL = prep_pdb_inputs_fresh_HERE / 'outputs' / 'pdb_availability.csv'
META = prep_pdb_inputs_fresh_REPO / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
ANA = prep_pdb_inputs_fresh_REPO / 'dataset_update' / 'main_plus_r2_r3_analysis_v12_corrected.csv'
prep_pdb_inputs_fresh_OUT = prep_pdb_inputs_fresh_HERE / 'outputs'
prep_pdb_inputs_fresh_PARTIAL = prep_pdb_inputs_fresh_OUT / 'pdb_scoring_inputs.partial.csv'
COVERAGE_MIN = 0.5
METHODS = {'X-ray'}
def prep_pdb_inputs_fresh_process(row):
    (e, pdb_id, chain, uni) = (row['Entry'], str(row['best_pdb_id']).strip().upper(), str(row['best_chain']).strip(), row['sequence'])
    out = {'Entry': e, 'pdb_id': pdb_id, 'pdb_chain': chain, 'fail_reason': None}
    try:
        cif = CIF_CACHE / f'{pdb_id}.cif'
        if not cif.exists():
            rcsb.fetch(pdb_id, 'cif', str(CIF_CACHE))
        if cif.stat().st_size > MAX_CIF_MB * 1000000.0:
            out['fail_reason'] = 'large_assembly_skipped'
            return out
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif)), model=1)
        aa = arr[struc.filter_amino_acids(arr)]
        sel = aa[aa.chain_id == chain]
        if sel.array_length() == 0:
            best = (None, 0.0, None)
            for ch in sorted(set(aa.chain_id)):
                ca = aa[(aa.chain_id == ch) & (aa.atom_name == 'CA')]
                if ca.array_length() == 0:
                    continue
                s = ''.join((three_to_one(r) for r in ca.res_name))
                (_, idn) = best_alignment(uni, s)
                if idn > best[1]:
                    best = (ch, idn, s)
            chain = best[0]
            sel = aa[aa.chain_id == chain]
            out['pdb_chain'] = chain
        ca = sel[sel.atom_name == 'CA']
        seq = ''.join((three_to_one(r) for r in ca.res_name))
        (off, idn) = best_alignment(uni, seq)
        out.update(pdb_identity=round(idn, 4), offset=off, chain_len=len(seq), uniprot_len=len(uni), chain_seq=seq)
        if idn < IDENTITY_MIN or 'X' in seq:
            out['fail_reason'] = f'identity {idn:.2f}' if idn < IDENTITY_MIN else 'nonstandard_X'
            return out
        out['orig_chain'] = chain
        sel = sel.copy()
        sel.chain_id[:] = 'A'
        cp = CHAIN_DIR / f'{e}_{pdb_id}_{chain}.pdb'
        pf = pdbio.PDBFile()
        pf.set_structure(sel)
        pf.write(str(cp))
        out['chain_pdb_path'] = str(cp)
        out['pdb_chain'] = 'A'
    except Exception as ex:
        out['fail_reason'] = f'error:{str(ex)[:40]}'
    return out
def prep_pdb_inputs_fresh_main():
    av = pd.read_csv(AVAIL)
    av = av[av.has_pdb & (av.coverage_frac >= COVERAGE_MIN) & av.best_method.isin(METHODS)].copy()
    seqs = pd.read_csv(META, low_memory=False).set_index('Entry')['sequence']
    av['sequence'] = av.Entry.map(seqs)
    av = av.dropna(subset=['sequence', 'best_pdb_id', 'best_chain'])
    print(f'Candidates (has_pdb, coverage>={COVERAGE_MIN}): {len(av)}')
    (done, results) = (set(), [])
    if prep_pdb_inputs_fresh_PARTIAL.exists():
        results = pd.read_csv(prep_pdb_inputs_fresh_PARTIAL).to_dict('records')
        done = {r['Entry'] for r in results}
    todo = av[~av.Entry.isin(done)].to_dict('records')
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(prep_pdb_inputs_fresh_process, r) for r in todo]
        for (i, f) in enumerate(tqdm(as_completed(futs), total=len(futs), desc='PDB chain prep')):
            results.append(f.result())
            if i % 50 == 0:
                pd.DataFrame(results).to_csv(prep_pdb_inputs_fresh_PARTIAL, index=False)
    info = pd.DataFrame(results).merge(av[['Entry', 'domain', 'coverage_frac', 'best_method', 'best_resolution_A']], on='Entry', how='left')
    ok = info[info.fail_reason.isna() & info.chain_pdb_path.notna()].copy()
    a = pd.read_csv(ANA, low_memory=False)[['Entry', 'species', 'protein_family'] + [c for c in SCORE_COLS]]
    final = ok.merge(a, on='Entry', how='left').rename(columns={'chain_seq': 'sequence'})
    keep = ['Entry', 'pdb_id', 'pdb_chain', 'sequence', 'chain_pdb_path', 'domain', 'species', 'protein_family', 'pdb_identity', 'offset', 'chain_len', 'uniprot_len', 'coverage_frac', 'best_method', 'best_resolution_A'] + [c for c in SCORE_COLS]
    final[[c for c in keep if c in final.columns]].to_csv(prep_pdb_inputs_fresh_OUT / 'pdb_scoring_inputs.csv', index=False)
    info.to_csv(prep_pdb_inputs_fresh_OUT / 'pdb_prep_diagnostics.csv', index=False)
    if prep_pdb_inputs_fresh_PARTIAL.exists():
        prep_pdb_inputs_fresh_PARTIAL.unlink()
    print(f'\nUsable PDB scoring inputs: {len(final)}')
    print('By domain:')
    print(final.domain.value_counts().to_string())
    print('By method:')
    print(final.best_method.value_counts().to_string())
    print('Excluded reasons:')
    print(info.fail_reason.value_counts(dropna=True).to_string())
    print(f'\nWrote pdb_scoring_inputs.csv (Entry/pdb_id/pdb_chain/sequence + chain_pdb_path)')
def prep_pdb_inputs_fresh__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(prep_pdb_inputs_fresh_HERE))
    prep_pdb_inputs_fresh_main()

# ---------- from build_pdb_cohort_features.py ----------
build_pdb_cohort_features_HERE = Path(__file__).resolve().parent
build_pdb_cohort_features_REPO = build_pdb_cohort_features_HERE.parent
build_pdb_cohort_features_COH = build_pdb_cohort_features_REPO / 'design' / 'outputs' / 'independent_cohort' / 'cohort_pdb_scoring_inputs.csv'
build_pdb_cohort_features_OUT = build_pdb_cohort_features_REPO / 'design' / 'outputs' / 'independent_cohort' / 'cohort_pdb_features.csv'
SEQ_FEATS = ['sequence_length', 'mw_per_residue', 'isoelectric_point', 'charge_at_ph7', 'acidic_residue_fraction', 'basic_residue_fraction', 'gravy', 'aromaticity', 'instability_index', 'proline_fraction', 'small_residue_fraction']
STRUCT_FEATS = ['ordered_percent', 'helix_sheet_contrast', 'rco', 'avg_cb_distance', 'surface_exposure']
CARRY = ['Entry', 'pdb_id', 'chain', 'domain', 'species_collapsed', 'protein_family', 'broad_function', 'resolution_A', 'resolved_len', 'seqres_len']
def clean_seq(s):
    """Match the v12 builder: U->C (selenocysteine), drop X."""
    if not isinstance(s, str):
        return ''
    return s.upper().replace('U', 'C').replace('X', '')
def build_pdb_cohort_features_main():
    df = pd.read_csv(build_pdb_cohort_features_COH)
    print(f'PDB cohort: {len(df)} chains')
    (seq_rows, struct_rows, seq_err, struct_err) = ([], [], 0, 0)
    for (_, r) in tqdm(df.iterrows(), total=len(df), desc='features'):
        cs = clean_seq(r['sequence'])
        try:
            f = sequence_features(cs) if cs else {}
            seq_rows.append({k: f.get(k) for k in SEQ_FEATS})
        except Exception:
            seq_rows.append({k: np.nan for k in SEQ_FEATS})
            seq_err += 1
        try:
            sfd = structure_features(r['chain_pdb_path'])
            struct_rows.append({k: sfd.get(k) for k in STRUCT_FEATS})
        except Exception as e:
            struct_rows.append({k: np.nan for k in STRUCT_FEATS})
            struct_err += 1
    seqf = pd.DataFrame(seq_rows)
    strf = pd.DataFrame(struct_rows)
    out = df[CARRY].reset_index(drop=True).copy()
    out['species'] = out['species_collapsed']
    for k in SEQ_FEATS:
        out[k] = seqf[k].values
    for k in STRUCT_FEATS:
        out[k] = strf[k].values
    build_pdb_cohort_features_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(build_pdb_cohort_features_OUT, index=False)
    print(f'\nWrote {build_pdb_cohort_features_OUT}  ({out.shape[0]} rows, {out.shape[1]} cols)')
    print(f'sequence-feature errors: {seq_err} | structure-feature errors: {struct_err}')
    miss = {k: int(out[k].isna().sum()) for k in MIXED_FEATURES if out[k].isna().sum()}
    print('Missing per feature:', miss or 'none')
    return out
def build_pdb_cohort_features__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(build_pdb_cohort_features_HERE))
    sys.path.insert(0, str(build_pdb_cohort_features_REPO))
    build_pdb_cohort_features_main()

# ---------- from score_esmif_cohort.py ----------
score_esmif_cohort_REPO = Path(__file__).resolve().parent.parent
score_esmif_cohort_COH = score_esmif_cohort_REPO / 'design' / 'outputs' / 'independent_cohort' / 'cohort_pdb_scoring_inputs.csv'
score_esmif_cohort_OUT = score_esmif_cohort_REPO / 'design' / 'outputs' / 'independent_cohort' / 'esmif_scores_pdb.csv'
def pdb_chain_id(path):
    """The extracted single-chain PDBs are relabelled to 'A'; read the actual
    chain letter from the file rather than trusting the original `chain` column."""
    with open(path) as fh:
        for line in fh:
            if line.startswith(('ATOM', 'HETATM')):
                return line[21]
    return 'A'
def score_esmif_cohort_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='score only first N (smoke test)')
    args = ap.parse_args()
    df = pd.read_csv(score_esmif_cohort_COH)
    df['Entry'] = df['Entry'].astype(str)
    done = set()
    if score_esmif_cohort_OUT.exists():
        prev = pd.read_csv(score_esmif_cohort_OUT)
        done = set(prev['Entry'].astype(str))
        print(f'resume: {len(done)} already scored')
    todo = df[~df['Entry'].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f'to score: {len(todo)} / {len(df)}')
    device = 'cpu'
    print('loading esm_if1_gvp4_t16_142M_UR50 ...')
    (model, alphabet) = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    write_header = not score_esmif_cohort_OUT.exists()
    n_ok = n_err = 0
    for (_, r) in tqdm(todo.iterrows(), total=len(todo), desc='ESM-IF'):
        rec = {'Entry': r['Entry'], 'pdb_id': r.get('pdb_id'), 'chain': r.get('chain'), 'esmif_score': np.nan, 'esmif_ll_withcoord': np.nan, 'scored_length': np.nan, 'valid_positions': np.nan, 'error': ''}
        try:
            chain_in_file = pdb_chain_id(r['chain_pdb_path'])
            (coords, _) = esm.inverse_folding.util.load_coords(r['chain_pdb_path'], chain_in_file)
            seq = str(r['sequence'])
            (ll_fullseq, ll_withcoord) = esm.inverse_folding.util.score_sequence(model, alphabet, coords, seq)
            coord_mask = np.all(np.isfinite(coords), axis=(-1, -2))
            rec.update(esmif_score=float(ll_fullseq), esmif_ll_withcoord=float(ll_withcoord), scored_length=len(seq), valid_positions=int(coord_mask.sum()))
            n_ok += 1
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
            n_err += 1
        pd.DataFrame([rec]).to_csv(score_esmif_cohort_OUT, mode='a', header=write_header, index=False)
        write_header = False
    print(f'\ndone: {n_ok} ok, {n_err} errors -> {score_esmif_cohort_OUT}')
def score_esmif_cohort__entry():
    score_esmif_cohort_main()

# ---------- from merge_pdb_cohort_scores.py ----------
merge_pdb_cohort_scores_REPO = Path(__file__).resolve().parent.parent
FEATURES = merge_pdb_cohort_scores_REPO / 'design' / 'outputs' / 'independent_cohort' / 'cohort_pdb_features.csv'
merge_pdb_cohort_scores_OUT = merge_pdb_cohort_scores_REPO / 'design' / 'outputs' / 'independent_cohort' / 'cohort_pdb_scored.csv'
DEFAULT_SCORES = Path.home() / 'Downloads' / 'PDB_cohort_scores'
RENAME = {'proteinmpnn': 'proteinmpnn_score', 'solublempnn': 'solublempnn_score', 'AlkalineMPNN': 'AlkalineMPNN_score', 'AcidophileMPNN': 'AcidophileMPNN_score', 'caliby_score': 'caliby_score', 'soluble_caliby_score': 'soluble_caliby_score', 'esmif_score': 'esmif_score', 'triflow_score': 'triflow_score', 'esm3_struct_cond_score': 'esm3_struct_cond_score', 'esm3_seq_only_score': 'esm3_seq_only_score', 'mif_score': 'mif_score', 'mifst_score': 'mifst_score', 'ESM2_15B_pppl_score': 'ESM2_15B_pppl_score', 'carp_640M_score': 'carp_640M_score', 'progen2_XL_score': 'progen2_XL_score', 'protgpt2_score': 'protgpt2_score'}
PANEL = ['proteinmpnn_score', 'solublempnn_score', 'caliby_score', 'soluble_caliby_score', 'esmif_score', 'triflow_score', 'esm3_struct_cond_score', 'mif_score', 'mifst_score', 'esm3_seq_only_score', 'ESM2_15B_pppl_score', 'carp_640M_score', 'progen2_XL_score', 'protgpt2_score']
FINETUNES = ['AlkalineMPNN_score', 'AcidophileMPNN_score']
def merge_pdb_cohort_scores_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores-dir', default=str(DEFAULT_SCORES))
    args = ap.parse_args()
    scores_dir = Path(args.scores_dir)
    master = pd.read_csv(FEATURES)
    master['Entry'] = master['Entry'].astype(str)
    cohort_ids = set(master['Entry'])
    print(f'features base: {len(master)} rows ({len(cohort_ids)} cohort Entries)')
    filled = {}
    for f in sorted(glob.glob(str(scores_dir / '*.csv'))):
        df = pd.read_csv(f)
        if 'Entry' not in df.columns:
            continue
        df['Entry'] = df['Entry'].astype(str)
        for (src, canon) in RENAME.items():
            if src not in df.columns:
                continue
            sub = df[['Entry', src]].dropna(subset=['Entry']).drop_duplicates('Entry')
            sub = sub[sub['Entry'].isin(cohort_ids)]
            if canon in filled:
                prev = master[['Entry', canon]].dropna()
                chk = prev.merge(sub.rename(columns={src: canon + '_new'}), on='Entry')
                if len(chk):
                    diff = (chk[canon] - chk[canon + '_new']).abs()
                    note = 'identical' if diff.max() < 1e-06 else f'DIFFERS max={diff.max():.4g}'
                    print(f'  dup {canon}: {os.path.basename(f)} ({note}) - kept {filled[canon]}')
                continue
            master = master.merge(sub.rename(columns={src: canon}), on='Entry', how='left')
            filled[canon] = os.path.basename(f)
    master.to_csv(merge_pdb_cohort_scores_OUT, index=False)
    print(f'\nWrote {merge_pdb_cohort_scores_OUT}  ({master.shape[0]} rows, {master.shape[1]} cols)')
    print('\nPanel coverage (non-null / 876):')
    (have, missing) = ([], [])
    for c in PANEL:
        if c in master.columns:
            n = int(master[c].notna().sum())
            have.append(c)
            print(f"  {c:26} {n:4d}/876   [{filled.get(c, '?')}]")
        else:
            missing.append(c)
    for c in FINETUNES:
        if c in master.columns:
            print(f"  + {c:26} {int(master[c].notna().sum()):4d}/876   [{filled.get(c, '?')}]  (fine-tune)")
    if missing:
        print('\n  ❌ STILL MISSING (need scoring):')
        for c in missing:
            print(f'     - {c}')
    print(f'\n{len(have)}/{len(PANEL)} panel models present.')
def merge_pdb_cohort_scores__entry():
    merge_pdb_cohort_scores_main()

_STEPS = {
    'prep-independent-cohort-structures': prep_independent_cohort_structures__entry,
    'prep-pdb-inputs-fresh': prep_pdb_inputs_fresh__entry,
    'build-pdb-cohort-features': build_pdb_cohort_features__entry,
    'score-esmif-cohort': score_esmif_cohort__entry,
    'merge-pdb-cohort-scores': merge_pdb_cohort_scores__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

