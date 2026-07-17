"""decoding_bias.finetune.design_inputs -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - build_design_acidbase_inputs
  - build_design_ph_axis_inputs
"""

from __future__ import annotations

import argparse
import glob
import numpy as np
import pandas as pd
import shutil
import sys
from pathlib import Path
from surface_features_alkaline import one_structure
from decoding_bias.features.sequence_features import calculate_sequence_features

# ---------- from build_design_acidbase_inputs.py ----------
HERE = Path(__file__).resolve().parent
PH = HERE / 'outputs' / 'designs_ph_features.csv'
SURF = HERE / 'outputs' / 'designs_surface_features.csv'
WT_PDB_DIR = HERE / 'arc_downloads' / 'rank001_flat'
SURF_COLS = ['surface_acidic_fraction', 'surface_basic_fraction', 'surface_net_charge', 'surface_ionizable_fraction']
def surf_features_from_pdb(path):
    d = one_structure(path)
    return {'surface_acidic_fraction': d['surf_acidic'], 'surface_basic_fraction': d['surf_basic'], 'surface_net_charge': d['surf_net_KR_DE'], 'surface_ionizable_fraction': d['surf_acidic'] + d['surf_basic']}
def build_design_acidbase_inputs_main():
    ph = pd.read_csv(PH)
    surf = pd.read_csv(SURF)[['uniprot_id', 'model', 'sample_idx'] + SURF_COLS]
    designs = ph.merge(surf, on=['uniprot_id', 'model', 'sample_idx'], how='left')
    miss = designs[SURF_COLS[0]].isna().sum()
    print(f'designs: {len(designs)} rows, {miss} without surface features')
    wt_seq = ph.drop_duplicates('uniprot_id')[['uniprot_id', 'wt_sequence']]
    wt_pdb = {Path(p).name.split('__')[0]: p for p in glob.glob(str(WT_PDB_DIR / '*__WT_unrelaxed_rank_001*.pdb'))}
    rows = []
    for r in wt_seq.itertuples(index=False):
        rec = {'uniprot_id': r.uniprot_id, 'model': 'WT', 'sample_idx': np.nan, 'wt_sequence': r.wt_sequence, 'designed_sequence': r.wt_sequence}
        rec.update(ph_features(r.wt_sequence))
        p = wt_pdb.get(r.uniprot_id)
        rec.update(surf_features_from_pdb(p) if p else {c: np.nan for c in SURF_COLS})
        rows.append(rec)
    wt = pd.DataFrame(rows)
    print(f'WT rows: {len(wt)}, with fold {sum((u in wt_pdb for u in wt_seq.uniprot_id))}/{len(wt_seq)}')
    out = pd.concat([designs, wt[designs.columns]], ignore_index=True)
    shutil.copy(PH, PH.parent / 'designs_ph_features.backup_preacidbase.csv')
    out.to_csv(PH, index=False)
    print(f'\nwrote {PH}  ({len(out)} rows: {len(designs)} designs + {len(wt)} WT)')
    print('surface cols per model (mean surface_net_charge):')
    print(out.groupby('model')['surface_net_charge'].mean().round(4).to_string())
def build_design_acidbase_inputs__entry():
    sys.path.insert(0, str(HERE))
    build_design_acidbase_inputs_main()

# ---------- from build_design_ph_axis_inputs.py ----------
REPO = Path(__file__).resolve().parent.parent
PH_FEATURES = ['sequence_length', 'isoelectric_point', 'charge_at_ph7', 'charge_per_residue', 'buffer_capacity', 'acidic_residue_fraction', 'basic_residue_fraction', 'ionizable_residue_fraction']
DESIGN_META = ['uniprot_id', 'species', 'domain', 'rank_class', 'target_cell', 'model', 'soluble_variant', 'sample_idx', 'seed', 'temperature', 'model_score', 'score_type', 'structure_path']
WT_META = ['uniprot_id', 'species', 'domain', 'rank_class', 'target_cell']
def clean_sequence(seq: object) -> str:
    return str(seq).strip().upper().replace('*', '')
def ph_features(seq: str) -> dict[str, float]:
    raw = calculate_sequence_features(clean_sequence(seq))
    out = {'sequence_length': raw.get('sequence_length', np.nan), 'isoelectric_point': raw.get('isoelectric_point', np.nan), 'charge_at_ph7': raw.get('charge_at_ph7', raw.get('charge_at_pH7', np.nan)), 'charge_per_residue': raw.get('charge_per_residue', np.nan), 'buffer_capacity': raw.get('buffer_capacity', np.nan), 'acidic_residue_fraction': raw.get('acidic_residue_fraction', np.nan), 'basic_residue_fraction': raw.get('basic_residue_fraction', np.nan), 'ionizable_residue_fraction': raw.get('ionizable_residue_fraction', raw.get('ionizable_fraction', np.nan))}
    return out
def discover_design_csvs(design_dir: Path) -> list[Path]:
    candidates = sorted((Path(p) for p in glob.glob(str(design_dir / 'designs_*.csv'))))
    usable = []
    for path in candidates:
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if {'wt_sequence', 'designed_sequence', 'model', 'uniprot_id'}.issubset(cols):
            usable.append(path)
    return usable
def build_tables(csv_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path) for path in csv_paths]
    raw = pd.concat(frames, ignore_index=True)
    wt_conflicts = raw.groupby('uniprot_id')['wt_sequence'].nunique(dropna=False).loc[lambda s: s > 1]
    if not wt_conflicts.empty:
        raise ValueError('WT sequence conflicts for: ' + ', '.join(map(str, wt_conflicts.index[:10])))
    design_records = []
    for row in raw.itertuples(index=False):
        rec = {col: getattr(row, col) for col in DESIGN_META if hasattr(row, col)}
        rec['designed_sequence'] = clean_sequence(getattr(row, 'designed_sequence'))
        rec['wt_sequence'] = clean_sequence(getattr(row, 'wt_sequence'))
        rec.update(ph_features(rec['designed_sequence']))
        design_records.append(rec)
    designs = pd.DataFrame(design_records)
    wt_base = raw.sort_values(['uniprot_id', 'model', 'sample_idx']).drop_duplicates('uniprot_id').copy()
    wt_records = []
    for row in wt_base.itertuples(index=False):
        rec = {col: getattr(row, col) for col in WT_META if hasattr(row, col)}
        rec['wt_sequence'] = clean_sequence(getattr(row, 'wt_sequence'))
        rec.update(ph_features(rec['wt_sequence']))
        wt_records.append(rec)
    wt = pd.DataFrame(wt_records)
    ordered_design_cols = [col for col in DESIGN_META + ['wt_sequence', 'designed_sequence'] + PH_FEATURES if col in designs.columns]
    ordered_wt_cols = [col for col in WT_META + ['wt_sequence'] + PH_FEATURES if col in wt.columns]
    return (designs[ordered_design_cols], wt[ordered_wt_cols])
def build_design_ph_axis_inputs_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-dir', default='/Users/lauradillon/Downloads/Designs', help='Directory containing designs_*.csv files.')
    parser.add_argument('--out-dir', default=None, help='Output directory. Defaults to <design-dir>/ph_axis_features.')
    args = parser.parse_args()
    design_dir = Path(args.design_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else design_dir / 'ph_axis_features'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = discover_design_csvs(design_dir)
    if not csv_paths:
        raise FileNotFoundError(f'No raw designs_*.csv files found in {design_dir}')
    (designs, wt) = build_tables(csv_paths)
    designs_path = out_dir / 'designs_ph_features.csv'
    wt_path = out_dir / 'wt_ph_features.csv'
    designs.to_csv(designs_path, index=False)
    wt.to_csv(wt_path, index=False)
    summary = designs.groupby('model', as_index=False).agg(n_designs=('uniprot_id', 'size'), n_wt=('uniprot_id', 'nunique'), mean_pI=('isoelectric_point', 'mean'), mean_charge_at_ph7=('charge_at_ph7', 'mean'), mean_acidic_fraction=('acidic_residue_fraction', 'mean'), mean_basic_fraction=('basic_residue_fraction', 'mean'))
    summary_path = out_dir / 'designs_ph_feature_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'Read {len(csv_paths)} design CSVs:')
    for path in csv_paths:
        print(f'  - {path}')
    print(f'Wrote {designs_path} ({len(designs)} designs)')
    print(f'Wrote {wt_path} ({len(wt)} WTs)')
    print(f'Wrote {summary_path}')
    print(summary.to_string(index=False))
def build_design_ph_axis_inputs__entry():
    sys.path.insert(0, str(REPO))
    build_design_ph_axis_inputs_main()

_STEPS = {
    'build-design-acidbase-inputs': build_design_acidbase_inputs__entry,
    'build-design-ph-axis-inputs': build_design_ph_axis_inputs__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

