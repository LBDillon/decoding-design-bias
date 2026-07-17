"""decoding_bias.design.design_folding -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - make_fold_fasta
  - build_fold_structure_manifest
  - extract_design_features
  - compute_design_surface_charge
"""

from __future__ import annotations

import argparse
import glob
import os
import pandas as pd
import re
import sys
from pathlib import Path
from tqdm import tqdm
from features_for_designs import compute_mixed_features, MIXED_FEATURES
from concurrent.futures import ProcessPoolExecutor, as_completed
from surface_features_alkaline import one_structure

# ---------- from make_fold_fasta.py ----------
make_fold_fasta_HERE = Path(__file__).resolve().parent
def sanitize(s):
    return re.sub('[^A-Za-z0-9_.-]', '-', str(s))
def make_fold_fasta_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dirs', nargs='+', default=[str(Path.home() / 'Downloads'), 'outputs'], help='dirs to scan for designs_*.csv')
    ap.add_argument('--exclude', nargs='*', default=['designs_MIF.csv'], help='filenames to skip (e.g. the broken pre-fix MIF)')
    ap.add_argument('--out', default='outputs/all_to_fold.fasta')
    args = ap.parse_args()
    csvs = []
    for d in args.csv_dirs:
        csvs += glob.glob(os.path.join(os.path.expanduser(d), 'designs_*.csv'))
    csvs = [c for c in csvs if os.path.basename(c) not in args.exclude]
    print('Design CSVs:', [os.path.basename(c) for c in csvs])
    (rows, seen) = ([], set())
    for c in csvs:
        df = pd.read_csv(c)
        for r in df.itertuples():
            key = (r.uniprot_id, r.model, r.sample_idx)
            if key in seen:
                continue
            seen.add(key)
            did = f'{sanitize(r.uniprot_id)}__{sanitize(r.model)}__s{r.sample_idx}'
            rows.append({'fold_id': did, 'uniprot_id': r.uniprot_id, 'model': r.model, 'sample_idx': r.sample_idx, 'domain': r.domain, 'rank_class': r.rank_class, 'is_wt': False, 'sequence': r.designed_sequence})
    inp = pd.read_csv(make_fold_fasta_HERE / 'design_input_proteins.csv')
    for r in inp.itertuples():
        did = f'{sanitize(r.uniprot_id)}__WT'
        rows.append({'fold_id': did, 'uniprot_id': r.uniprot_id, 'model': 'WT', 'sample_idx': -1, 'domain': r.domain, 'rank_class': r.rank_class, 'is_wt': True, 'sequence': r.wt_sequence})
    man = pd.DataFrame(rows)
    out_fa = make_fold_fasta_HERE / args.out
    out_fa.parent.mkdir(exist_ok=True)
    with open(out_fa, 'w') as fh:
        for r in man.itertuples():
            fh.write(f'>{r.fold_id}\n{r.sequence}\n')
    man.drop(columns=['sequence']).to_csv(out_fa.with_suffix('.manifest.csv'), index=False)
    print(f'\nWrote {out_fa}')
    print(f'  {len(man)} sequences  ({(~man.is_wt).sum()} designs + {man.is_wt.sum()} WTs)')
    print(f'  by model:\n{man.model.value_counts().to_string()}')
    print(f"Manifest: {out_fa.with_suffix('.manifest.csv')}")
def make_fold_fasta__entry():
    make_fold_fasta_main()

# ---------- from build_fold_structure_manifest.py ----------
build_fold_structure_manifest_HERE = Path(__file__).resolve().parent
REPO = build_fold_structure_manifest_HERE.parent
def rel_to_design(path: Path, *, resolve: bool=False) -> str:
    p = path.resolve() if resolve else path
    return os.path.relpath(p, build_fold_structure_manifest_HERE)
def fold_id_from_pdb_name(name: str) -> str | None:
    for marker in ('_unrelaxed_rank_001', '_relaxed_rank_001', '_rank_001'):
        if marker in name:
            return name.split(marker, 1)[0]
    return None
def expected_fold_id(row: pd.Series) -> str:
    if bool(row['is_wt']):
        return f"{row['uniprot_id']}__WT"
    sample_idx = int(row['design_number']) - 1
    return f"{row['uniprot_id']}__{row['model']}__s{sample_idx}"
def scan_rank1_pdbs(roots: list[Path]) -> tuple[dict[str, Path], pd.DataFrame]:
    """Return one preferred PDB per fold_id, plus a full scan table."""
    rows = []
    preferred: dict[str, Path] = {}
    for root in roots:
        root = root.expanduser()
        if not root.is_absolute():
            root = build_fold_structure_manifest_HERE / root
        if not root.exists():
            continue
        for pdb in sorted(root.rglob('*rank_001*.pdb')):
            fold_id = fold_id_from_pdb_name(pdb.name)
            if fold_id is None:
                continue
            resolved = pdb.resolve()
            rows.append({'fold_id': fold_id, 'pdb_path': rel_to_design(resolved), 'pdb_abs_path': str(resolved), 'pdb_file': pdb.name, 'source_root': rel_to_design(root)})
            preferred.setdefault(fold_id, resolved)
    scan = pd.DataFrame(rows)
    return (preferred, scan)
def make_flat_links(mapping: dict[str, Path], flat_dir: Path) -> dict[str, Path]:
    flat_dir.mkdir(parents=True, exist_ok=True)
    linked: dict[str, Path] = {}
    for (fold_id, target) in sorted(mapping.items()):
        link = flat_dir / target.name
        if link.exists() or link.is_symlink():
            if link.resolve() == target.resolve():
                linked[fold_id] = link
                continue
            link.unlink()
        rel_target = os.path.relpath(target.resolve(), flat_dir.resolve())
        link.symlink_to(rel_target)
        linked[fold_id] = link
    return linked
def build_fold_structure_manifest_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--design-csv', default='outputs/all_designs_and_wt.csv')
    ap.add_argument('--fold-manifest', default='outputs/all_to_fold.manifest.csv')
    ap.add_argument('--pdb-roots', nargs='+', default=['arc_downloads/colabfold_chunks_home', 'arc_downloads/colabfold_retry', 'arc_downloads/colabfold_chunks', 'arc_downloads/rank001_all'], help='Downloaded rank-1 PDB directories, relative to design/ unless absolute.')
    ap.add_argument('--flat-dir', default='arc_downloads/rank001_flat')
    ap.add_argument('--out', default='outputs/all_designs_and_wt_with_folds.csv')
    ap.add_argument('--structure-manifest', default='outputs/colabfold_rank1_manifest.csv')
    args = ap.parse_args()
    design_csv = build_fold_structure_manifest_HERE / args.design_csv
    fold_manifest = build_fold_structure_manifest_HERE / args.fold_manifest
    out = build_fold_structure_manifest_HERE / args.out
    structure_manifest = build_fold_structure_manifest_HERE / args.structure_manifest
    flat_dir = build_fold_structure_manifest_HERE / args.flat_dir
    designs = pd.read_csv(design_csv)
    folds = pd.read_csv(fold_manifest)
    designs['fold_id'] = designs.apply(expected_fold_id, axis=1)
    manifest_ids = set(folds['fold_id'])
    missing_from_manifest = sorted(set(designs['fold_id']) - manifest_ids)
    if missing_from_manifest:
        raise SystemExit(f'{len(missing_from_manifest)} derived fold_ids are absent from {fold_manifest}: {missing_from_manifest[:5]}')
    pdb_roots = [Path(p) for p in args.pdb_roots]
    (pdb_by_fold, scan) = scan_rank1_pdbs(pdb_roots)
    links_by_fold = make_flat_links(pdb_by_fold, flat_dir)
    designs['has_colabfold_rank1'] = designs['fold_id'].isin(pdb_by_fold)
    designs['colabfold_rank1_pdb'] = designs['fold_id'].map(lambda fid: rel_to_design(pdb_by_fold[fid]) if fid in pdb_by_fold else '')
    designs['colabfold_rank1_pdb_abs'] = designs['fold_id'].map(lambda fid: str(pdb_by_fold[fid]) if fid in pdb_by_fold else '')
    designs['colabfold_rank1_flat_pdb'] = designs['fold_id'].map(lambda fid: rel_to_design(links_by_fold[fid]) if fid in links_by_fold else '')
    out.parent.mkdir(parents=True, exist_ok=True)
    designs.to_csv(out, index=False)
    if scan.empty:
        scan = pd.DataFrame(columns=['fold_id', 'pdb_path', 'pdb_abs_path', 'pdb_file', 'source_root'])
    scan['is_preferred'] = scan.apply(lambda r: r['fold_id'] in pdb_by_fold and r['pdb_abs_path'] == str(pdb_by_fold[r['fold_id']]), axis=1)
    scan.to_csv(structure_manifest, index=False)
    n_expected = len(designs)
    n_found = int(designs['has_colabfold_rank1'].sum())
    n_designs = int((~designs['is_wt']).sum())
    n_designs_found = int((designs['has_colabfold_rank1'] & ~designs['is_wt']).sum())
    n_wt = int(designs['is_wt'].sum())
    n_wt_found = int((designs['has_colabfold_rank1'] & designs['is_wt']).sum())
    dupes = int(scan.duplicated('fold_id').sum()) if not scan.empty else 0
    print(f'Expected rows: {n_expected}')
    print(f'Rank-1 structures linked: {n_found}/{n_expected}')
    print(f'  designs: {n_designs_found}/{n_designs}')
    print(f'  WTs:     {n_wt_found}/{n_wt}')
    print(f'Duplicate rank-1 PDB entries seen: {dupes}')
    print(f'Wrote: {out}')
    print(f'Wrote: {structure_manifest}')
    print(f'Flat symlinks: {flat_dir} ({len(links_by_fold)} links)')
def build_fold_structure_manifest__entry():
    build_fold_structure_manifest_main()

# ---------- from extract_design_features.py ----------
def read_fasta(path):
    (seqs, h) = ({}, None)
    for line in open(path):
        line = line.rstrip()
        if line.startswith('>'):
            h = line[1:].split()[0]
            seqs[h] = ''
        elif h:
            seqs[h] += line
    return seqs
def find_pdb(pdb_dir, fold_id):
    for pat in (f'{fold_id}_*rank_001*.pdb', f'{fold_id}_*rank_1*.pdb', f'{fold_id}*.pdb'):
        hits = sorted(glob.glob(os.path.join(pdb_dir, pat)))
        if hits:
            return hits[0]
    return None
def extract_design_features_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb-dir', required=True)
    ap.add_argument('--fasta', default='outputs/all_to_fold.fasta')
    ap.add_argument('--manifest', default='outputs/all_to_fold.manifest.csv')
    ap.add_argument('--out-dir', default='outputs')
    args = ap.parse_args()
    seqs = read_fasta(args.fasta)
    man = pd.read_csv(args.manifest)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (rows, missing) = ([], [])
    for r in tqdm(list(man.itertuples()), desc='features'):
        pdb = find_pdb(args.pdb_dir, r.fold_id)
        if pdb is None or r.fold_id not in seqs:
            missing.append(r.fold_id)
            continue
        try:
            feats = compute_mixed_features(seqs[r.fold_id], pdb)
        except Exception as e:
            missing.append(f'{r.fold_id} ({e})')
            continue
        rec = {'uniprot_id': r.uniprot_id, 'model': r.model, 'sample_idx': r.sample_idx, 'domain': r.domain, 'rank_class': r.rank_class, 'is_wt': r.is_wt}
        rec.update(feats)
        rows.append(rec)
    allf = pd.DataFrame(rows)
    designs = allf[~allf.is_wt].drop(columns=['is_wt'])
    wt = allf[allf.is_wt].drop(columns=['is_wt', 'model', 'sample_idx'])
    designs.to_csv(out / 'designs_features.csv', index=False)
    wt.to_csv(out / 'wt_features.csv', index=False)
    print(f"\nWrote {out / 'designs_features.csv'} ({len(designs)} designs)")
    print(f"Wrote {out / 'wt_features.csv'} ({len(wt)} WTs)")
    if missing:
        print(f'\n[WARN] {len(missing)} structures missing/failed:', missing[:8], '…' if len(missing) > 8 else '')
    miss_feat = {k: int(designs[k].isna().sum()) for k in MIXED_FEATURES if designs[k].isna().sum()}
    print('Missing feature values (designs):', miss_feat or 'none')
def extract_design_features__entry():
    extract_design_features_main()

# ---------- from compute_design_surface_charge.py ----------
compute_design_surface_charge_HERE = Path(__file__).resolve().parent
PDB_DIRS = [compute_design_surface_charge_HERE / 'arc_downloads' / 'rank001_flat', compute_design_surface_charge_HERE / 'outputs' / 'colabfold_out_ft', compute_design_surface_charge_HERE / 'outputs' / 'colabfold_out_ft020']
OUT = compute_design_surface_charge_HERE / 'outputs' / 'designs_surface_features.csv'
FID = re.compile('(.+?)__(.+?)__s(\\d+)$')
def fold_id(p):
    return Path(p).name.split('_unrelaxed_rank')[0]
def worker(p):
    fid = fold_id(p)
    if fid.endswith('__WT'):
        return None
    m = FID.match(fid)
    if not m:
        return None
    (uni, model, s) = (m.group(1), m.group(2), int(m.group(3)))
    try:
        d = one_structure(p)
    except Exception as ex:
        return dict(uniprot_id=uni, model=model, sample_idx=s, err=str(ex)[:60])
    return dict(uniprot_id=uni, model=model, sample_idx=s, surface_acidic_fraction=d['surf_acidic'], surface_basic_fraction=d['surf_basic'], surface_net_charge=d['surf_net_KR_DE'], surface_ionizable_fraction=d['surf_acidic'] + d['surf_basic'], n_surf=d['surf_n'])
def compute_design_surface_charge_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merge', action='store_true', help='left-join the 4 features into design/outputs/designs_features.csv')
    args = ap.parse_args()
    pdbs = []
    for d in PDB_DIRS:
        hits = sorted(glob.glob(str(d / '*rank_001*.pdb')))
        pdbs += hits
        print(f'{len(hits):>5}  {d}')
    print(f'{len(pdbs):>5}  total rank-1 PDBs\n')
    (rows, errs) = ([], [])
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(worker, p) for p in pdbs]
        for (i, fut) in enumerate(as_completed(futs)):
            r = fut.result()
            if r is None:
                continue
            (errs if 'err' in r else rows).append(r)
            if i % 400 == 0:
                print(f'  {i}/{len(pdbs)}', flush=True)
    df = pd.DataFrame(rows).sort_values(['model', 'uniprot_id', 'sample_idx']).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}  ({len(df)} designs, {len(errs)} errors)')
    print('\nper-model surface_net_charge / acidic / basic / ionizable (mean):')
    print(df.groupby('model')[['surface_net_charge', 'surface_acidic_fraction', 'surface_basic_fraction', 'surface_ionizable_fraction']].mean().round(4).to_string())
    if errs:
        print('\nerrors (first few):', errs[:3])
    if args.merge:
        feats = ['surface_acidic_fraction', 'surface_basic_fraction', 'surface_net_charge', 'surface_ionizable_fraction']
        dfm = pd.read_csv(compute_design_surface_charge_HERE / 'outputs' / 'designs_features.csv')
        dfm = dfm.drop(columns=[c for c in feats if c in dfm.columns], errors='ignore')
        merged = dfm.merge(df[['uniprot_id', 'model', 'sample_idx'] + feats], on=['uniprot_id', 'model', 'sample_idx'], how='left')
        miss = merged[feats[0]].isna().sum()
        merged.to_csv(compute_design_surface_charge_HERE / 'outputs' / 'designs_features.csv', index=False)
        print(f'\nmerged into designs_features.csv ({len(merged)} rows, {miss} unmatched)')
def compute_design_surface_charge__entry():
    sys.path.insert(0, str(compute_design_surface_charge_HERE))
    compute_design_surface_charge_main()

_STEPS = {
    'make-fold-fasta': make_fold_fasta__entry,
    'build-fold-structure-manifest': build_fold_structure_manifest__entry,
    'extract-design-features': extract_design_features__entry,
    'compute-design-surface-charge': compute_design_surface_charge__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

