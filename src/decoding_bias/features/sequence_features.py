"""
Sequence-derived biochemical property calculation.

Calculates features from protein sequences using BioPython's ProteinAnalysis,
matching Table 6 in the paper.

Usage:
    python -m src.features.sequence_features --input data.csv --output features.csv
"""

import argparse
import os
import time
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)


def calculate_sequence_features(seq):
    """
    Calculate biochemical properties for a protein sequence.

    Args:
        seq: Amino acid sequence string (single-letter code).

    Returns:
        Dictionary of computed features.
    """
    try:
        sequence_length = len(seq)
        if sequence_length == 0:
            raise ValueError("Empty sequence")

        analysis = ProteinAnalysis(seq)
        if hasattr(analysis, 'get_amino_acids_percent'):
            aa_counts = analysis.get_amino_acids_percent()
        else:
            aa_counts = analysis.amino_acids_percent
        if aa_counts and max(aa_counts.values()) > 1.0:
            aa_counts = {aa: pct / 100.0 for aa, pct in aa_counts.items()}

        acidic = ['D', 'E']
        basic = ['K', 'R', 'H']
        aromatic = ['F', 'W', 'Y']
        ionizable = ['D', 'E', 'K', 'R', 'H', 'C', 'Y']
        small = ['A', 'G', 'S', 'T']
        hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']

        acidic_fraction = sum(aa_counts.get(aa, 0) for aa in acidic)
        basic_fraction = sum(aa_counts.get(aa, 0) for aa in basic)
        aromatic_fraction = sum(aa_counts.get(aa, 0) for aa in aromatic)
        ionizable_fraction = sum(aa_counts.get(aa, 0) for aa in ionizable)
        proline_fraction = aa_counts.get('P', 0)
        small_residue_fraction = sum(aa_counts.get(aa, 0) for aa in small)
        hydrophobic_fraction = sum(aa_counts.get(aa, 0) for aa in hydrophobic)

        acidic_count = sum(seq.count(aa) for aa in acidic)
        basic_count = sum(seq.count(aa) for aa in basic)

        feats = {
            'sequence_length': sequence_length,
            'molecular_weight': analysis.molecular_weight(),
            'aromaticity': analysis.aromaticity(),
            'instability_index': analysis.instability_index(),
            'isoelectric_point': analysis.isoelectric_point(),
            'charge_at_pH7': analysis.charge_at_pH(7.0),
            'gravy': analysis.gravy(),
        }

        feats.update({
            'acidic_fraction': acidic_fraction,
            'basic_fraction': basic_fraction,
            'aromatic_fraction': aromatic_fraction,
            'ionizable_fraction': ionizable_fraction,
            'proline_fraction': proline_fraction,
            'small_residue_fraction': small_residue_fraction,
            'hydrophobic_fraction': hydrophobic_fraction,
        })

        feats['mw_per_residue'] = feats['molecular_weight'] / sequence_length
        feats['charge_per_residue'] = feats['charge_at_pH7'] / sequence_length

        charge_at_pH6 = analysis.charge_at_pH(6.0)
        charge_at_pH8 = analysis.charge_at_pH(8.0)
        feats['buffer_capacity'] = abs(charge_at_pH8 - charge_at_pH6) / 2.0

        if basic_count > 0:
            feats['acidic_basic_ratio'] = acidic_count / basic_count
        else:
            feats['acidic_basic_ratio'] = np.inf if acidic_count > 0 else 0.0

        feats['charge_asymmetry'] = abs(basic_count - acidic_count) / sequence_length
        feats['basic_residue_fraction'] = basic_fraction
        feats['acidic_residue_fraction'] = acidic_fraction

        return feats

    except Exception as e:
        logger.error(f"Error calculating features for sequence: {e}")
        feature_names = [
            'sequence_length', 'molecular_weight', 'aromaticity', 'instability_index',
            'isoelectric_point', 'charge_at_pH7', 'gravy',
            'acidic_fraction', 'basic_fraction', 'aromatic_fraction', 'ionizable_fraction',
            'proline_fraction', 'small_residue_fraction', 'hydrophobic_fraction',
            'mw_per_residue', 'charge_per_residue', 'buffer_capacity',
            'acidic_basic_ratio', 'charge_asymmetry',
            'basic_residue_fraction', 'acidic_residue_fraction'
        ]
        return {k: np.nan for k in feature_names}


def standardize_features(df, feature_cols):
    """
    Add z-score standardized versions of specified features.

    Args:
        df: DataFrame containing the feature columns.
        feature_cols: List of column names to standardize.

    Returns:
        DataFrame with added z-score columns (suffixed '_zscore').
    """
    standardized_cols = {}

    for col in feature_cols:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if len(values) < 2 or values.std() < 1e-10:
            logger.warning(f"Skipping standardization for {col}: insufficient variance")
            continue
        mean_val = df[col].mean()
        std_val = df[col].std()
        standardized_cols[f'{col}_zscore'] = (df[col] - mean_val) / std_val

    for col_name, col_data in standardized_cols.items():
        df[col_name] = col_data

    logger.info(f"Standardized {len(standardized_cols)} features")
    return df


def main(csv_path, output_path, sequence_col='sequence', id_col='Entry'):
    """
    Calculate sequence features for all proteins in a CSV file.

    Args:
        csv_path: Path to input CSV with protein sequences.
        output_path: Path for output CSV with computed features.
        sequence_col: Name of the column containing sequences.
        id_col: Name of the column containing protein identifiers.
    """
    meta_df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(meta_df)} entries from {csv_path}")

    if sequence_col not in meta_df.columns:
        raise ValueError(f"Input CSV must contain '{sequence_col}' column")

    start = time.time()
    records = []
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc='Calculating features'):
        seq = str(row[sequence_col])
        feats = calculate_sequence_features(seq)
        feats[id_col] = row.get(id_col, idx)
        records.append(feats)

    logger.info(f"Processed {len(records)} sequences in {(time.time() - start):.1f}s")

    feat_df = pd.DataFrame(records)

    overlapping_cols = set(meta_df.columns) & set(feat_df.columns) - {id_col}
    if overlapping_cols:
        logger.warning(f"Dropping {len(overlapping_cols)} overlapping columns from input: {overlapping_cols}")
        meta_df = meta_df.drop(columns=list(overlapping_cols))

    df = pd.merge(meta_df, feat_df, on=id_col, how='inner')

    features_to_standardize = [
        'molecular_weight', 'mw_per_residue', 'aromaticity', 'instability_index',
        'isoelectric_point', 'charge_at_pH7', 'charge_per_residue', 'gravy',
        'buffer_capacity', 'charge_asymmetry'
    ]
    df = standardize_features(df, features_to_standardize)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Features written to {output_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Calculate sequence-derived biochemical features")
    parser.add_argument("--input", required=True, help="Input CSV with protein sequences")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--sequence-col", default="sequence", help="Column name for sequences")
    parser.add_argument("--id-col", default="Entry", help="Column name for protein IDs")
    args = parser.parse_args()
    main(args.input, args.output, args.sequence_col, args.id_col)
