"""Merge ProteinMPNN scoring results with annotations needed for variance decomposition.

Inputs:
  - AF result CSV   (Entry, sequence_score, entropy, sequence_length, mean_confidence)
  - PDB result CSV  (same + pdb_id)

Annotation sources (in priority order):
  1. Decoding_Bias_Dataset_updated.csv  → full annotations (7,843 existing entries)
  2. expansion_for_scoring_monomer_FINAL.csv  → AF-mode metadata (sequence + classifications)
  3. expansion_for_scoring_monomer_PDB_FINAL.csv  → PDB-mode metadata
  4. UniProt REST API  → fallback for missing species / taxonomy

For expansion entries lacking computed biophysical features:
  - Sequence-derived features computed inline using BioPython ProtParam
  - Structural features (helix_percent, avg_cb_distance, rco, avg_plddt) flagged as
    'needs_structural_extraction' - must be computed from PDB/AF structures separately

Outputs:
  - <input_basename>_annotated.csv         columns ready for variance decomposition
  - <input_basename>_missing_features.csv  rows with incomplete annotations
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis

REPO_ROOT = Path(__file__).resolve().parent.parent
EXISTING = REPO_ROOT / 'dataset_update' / 'Decoding_Bias_Dataset_updated.csv'
EXP_AF = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_FINAL.csv'
EXP_PDB = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_PDB_FINAL.csv'

# Columns needed downstream for variance decomposition
NEEDED_COLS = [
    'species', 'domain', 'sequence', 'protein_name', 'protein_family', 'broad_function',
    'sequence_length', 'isoelectric_point', 'charge_at_ph7', 'gravy', 'instability_index',
    'mw_per_residue', 'aromaticity',
    'helix_percent', 'sheet_percent', 'surface_exposure', 'avg_cb_distance', 'rco',
    'avg_plddt', 'compactness',
]

SEQ_FEATURES = [
    'isoelectric_point', 'charge_at_ph7', 'gravy', 'instability_index',
    'mw_per_residue', 'aromaticity',
]
STRUCTURAL_FEATURES = [
    'helix_percent', 'sheet_percent', 'surface_exposure',
    'avg_cb_distance', 'rco', 'compactness',
]


# ---- Inline biophysical feature calculation -------------------------------

def compute_seq_features(seq: str) -> dict:
    """Sequence-derived features for one protein. Mirrors 02_calculate_sequence_features.py."""
    if not isinstance(seq, str) or not seq:
        return {f: None for f in SEQ_FEATURES + ['sequence_length']}
    seq = seq.upper().replace('U', 'C').replace('X', '')  # selenocysteine→cys, drop unknowns
    if not seq:
        return {f: None for f in SEQ_FEATURES + ['sequence_length']}
    try:
        a = ProteinAnalysis(seq)
        L = len(seq)
        return {
            'sequence_length': L,
            'mw_per_residue': a.molecular_weight() / L,
            'aromaticity': a.aromaticity(),
            'instability_index': a.instability_index(),
            'isoelectric_point': a.isoelectric_point(),
            'charge_at_ph7': a.charge_at_pH(7.0),
            'gravy': a.gravy(),
        }
    except Exception:
        return {f: None for f in SEQ_FEATURES + ['sequence_length']}


# ---- UniProt fallback -----------------------------------------------------

def fetch_uniprot_taxa(entries: list[str]) -> dict[str, dict]:
    """Query UniProt for species + taxonomic lineage for entries missing those.

    Returns dict mapping Entry → {species, domain, organism_full}.
    Uses the public REST endpoint (rate-limited; one request per entry).
    """
    out = {}
    for entry in entries:
        url = f"https://rest.uniprot.org/uniprotkb/{entry}.json"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                out[entry] = {}
                continue
            d = r.json()
            org = d.get('organism', {})
            sci = org.get('scientificName', '')
            lineage = org.get('lineage', [])
            domain = next((t for t in ('Viruses', 'Archaea', 'Bacteria', 'Eukaryota')
                          if t in lineage), 'Unknown')
            # Genus species (drop strain info)
            short = ' '.join(sci.split()[:2]) if sci else ''
            out[entry] = {'species': short, 'domain': domain, 'organism_full': sci}
        except requests.RequestException:
            out[entry] = {}
        time.sleep(0.05)  # be polite
    return out


# ---- Main merge logic -----------------------------------------------------

def annotate(scores_path: Path, out_dir: Path, label: str = '') -> None:
    print(f"\n=== Annotating {scores_path.name} ===")
    scores = pd.read_csv(scores_path)
    print(f"  loaded {len(scores)} rows")

    print("Loading reference datasets...")
    existing = pd.read_csv(EXISTING, low_memory=False)
    exp_af = pd.read_csv(EXP_AF, low_memory=False)
    exp_pdb = pd.read_csv(EXP_PDB, low_memory=False) if EXP_PDB.exists() else pd.DataFrame()

    annotated = scores.copy()
    annotated['source'] = None  # which reference each Entry came from

    # 1. Pull from existing dataset (has full annotations including structural)
    existing_keep = ['Entry'] + [c for c in NEEDED_COLS if c in existing.columns]
    ann_existing = existing[existing_keep].rename(columns={
        # in case of any column name mismatches, normalize here
    })
    annotated = annotated.merge(ann_existing, on='Entry', how='left', suffixes=('', '_ex'))
    matched_existing = annotated[NEEDED_COLS[0]].notna()
    annotated.loc[matched_existing, 'source'] = 'existing_dataset'
    print(f"  matched in existing dataset: {matched_existing.sum()}")

    # 2. For unmatched, pull from expansion CSVs (have classifications + metadata)
    unmatched_mask = annotated['source'].isna()
    if unmatched_mask.any():
        print(f"  unmatched after existing: {unmatched_mask.sum()}")
        for exp_df, label_src in [(exp_af, 'expansion_AF'), (exp_pdb, 'expansion_PDB')]:
            if exp_df.empty:
                continue
            exp_keep = ['Entry']
            for c in ['Organism', 'domain', 'sequence', 'protein_name_clean',
                      'protein_family', 'broad_function']:
                if c in exp_df.columns:
                    exp_keep.append(c)
            sub = exp_df[exp_keep].copy()
            sub = sub.rename(columns={
                'Organism': 'species_exp',
                'protein_name_clean': 'protein_name_exp',
                'sequence': 'sequence_exp',
                'domain': 'domain_exp',
                'protein_family': 'protein_family_exp',
                'broad_function': 'broad_function_exp',
            })
            annotated = annotated.merge(sub, on='Entry', how='left')

            for col_orig, col_new in [('species', 'species_exp'),
                                       ('domain', 'domain_exp'),
                                       ('sequence', 'sequence_exp'),
                                       ('protein_name', 'protein_name_exp'),
                                       ('protein_family', 'protein_family_exp'),
                                       ('broad_function', 'broad_function_exp')]:
                if col_new in annotated.columns:
                    fill = annotated[col_orig].isna() & annotated[col_new].notna()
                    annotated.loc[fill, col_orig] = annotated.loc[fill, col_new]
                    annotated.loc[fill, 'source'] = label_src
                    annotated.drop(columns=[col_new], inplace=True)

    # 3. Compute sequence-derived features for entries without them
    needs_seq_feat = annotated[SEQ_FEATURES[0]].isna() & annotated['sequence'].notna()
    if needs_seq_feat.any():
        print(f"  computing biophysical features for {needs_seq_feat.sum()} entries...")
        for idx in annotated[needs_seq_feat].index:
            feats = compute_seq_features(annotated.at[idx, 'sequence'])
            for k, v in feats.items():
                annotated.at[idx, k] = v

    # 4. Flag missing structural features
    missing_struct = annotated[STRUCTURAL_FEATURES[0]].isna()
    annotated['needs_structural_extraction'] = missing_struct
    annotated['needs_avg_plddt'] = annotated['avg_plddt'].isna()
    print(f"  needs structural extraction: {missing_struct.sum()}")
    print(f"  needs avg_plddt:             {annotated['needs_avg_plddt'].sum()}")

    # 5. Save outputs
    out_dir.mkdir(exist_ok=True)
    base = scores_path.stem
    out_main = out_dir / f"{base}_annotated.csv"
    annotated.to_csv(out_main, index=False)
    print(f"\nSaved annotated CSV: {out_main}")

    missing = annotated[
        annotated['species'].isna() | annotated['domain'].isna() |
        annotated['protein_family'].isna()
    ]
    if not missing.empty:
        out_miss = out_dir / f"{base}_missing_annotations.csv"
        missing[['Entry', 'species', 'domain', 'protein_family', 'source']].to_csv(out_miss, index=False)
        print(f"  warning: {len(missing)} rows missing species/domain/protein_family - see {out_miss.name}")
        print(f"  → consider running fetch_uniprot_taxa() on these")

    # Quick summary of what's downstream-analyzable
    full = annotated[
        annotated[['species', 'domain', 'protein_family',
                   'sequence_length', 'isoelectric_point', 'gravy']].notna().all(axis=1)
    ]
    print(f"\n  ready for VD (with seq features only):    {len(full):,} / {len(annotated):,}")
    full_struct = full[full[STRUCTURAL_FEATURES + ['avg_plddt']].notna().all(axis=1)]
    print(f"  ready for VD (with structural + pLDDT):   {len(full_struct):,} / {len(annotated):,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores_csv', type=Path, nargs='+',
                        help='ProteinMPNN result CSV(s) to annotate')
    parser.add_argument('--output-dir', type=Path,
                        default=REPO_ROOT / 'dataset_update' / 'scoring_results')
    args = parser.parse_args()

    for p in args.scores_csv:
        annotate(p, args.output_dir)


if __name__ == '__main__':
    main()
