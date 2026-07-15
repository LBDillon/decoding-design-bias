"""
Structure-derived feature extraction from PDB files.

Downloads AlphaFold models from the EBI database when local PDB files are
missing, then extracts geometric, secondary-structure, and pLDDT features.

Usage:
    python -m src.features.structural_features \
        --input data.csv --pdb-dir structures/ --output structural_features.csv
"""

import argparse
import os
import logging

import pandas as pd
import numpy as np
import requests
from Bio.PDB import MMCIFParser, PDBParser
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

AF_API_URL = 'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}'
AF_URLS = [
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb',
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.cif',
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb',
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif',
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v3.pdb',
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v2.pdb',
]


def scan_existing_structures(pdb_dir):
    """Scan PDB directory and return mapping of Entry IDs to PDB file paths."""
    if not os.path.exists(pdb_dir):
        logger.warning(f"PDB directory does not exist: {pdb_dir}")
        return {}

    pdb_files = {}
    pdb_path = Path(pdb_dir)
    structure_files = list(pdb_path.glob("*.pdb")) + list(pdb_path.glob("*.cif"))
    for structure_file in structure_files:
        filename = structure_file.stem
        pdb_files[filename] = str(structure_file)
        if filename.startswith('AF-') and '-F1-model' in filename:
            entry_id = filename.split('-')[1]
            pdb_files[entry_id] = str(structure_file)

    logger.info(f"Found {len(pdb_files)} existing structure files in {pdb_dir}")
    return pdb_files


def parse_structure(pdb_path):
    """Load a PDB file into a Bio.PDB Structure object."""
    try:
        suffix = Path(pdb_path).suffix.lower()
        parser = MMCIFParser(QUIET=True) if suffix == '.cif' else PDBParser(QUIET=True)
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
        if not structure or len(structure) == 0 or len(structure[0]) == 0:
            raise ValueError("Empty structure detected")
        return structure
    except Exception as e:
        logger.error(f"Failed to parse structure from {pdb_path}: {e}")
        return None


def _alphafold_download_urls(uniprot_id):
    """Return current AlphaFold URLs for an accession, preferring PDB files."""
    urls = []
    try:
        resp = requests.get(AF_API_URL.format(uniprot_id=uniprot_id), timeout=30)
        resp.raise_for_status()
        predictions = resp.json()
        if isinstance(predictions, list):
            def _prediction_rank(pred):
                accession = str(pred.get('uniprotAccession') or '')
                entry_id = str(pred.get('entryId') or '')
                exact = accession == uniprot_id or entry_id == f"AF-{uniprot_id}-F1"
                start = pred.get('uniprotStart') or 10**9
                end = pred.get('uniprotEnd') or 0
                return (0 if exact else 1, start, -end, entry_id)

            predictions = sorted(
                predictions,
                key=_prediction_rank,
            )
            for pred in predictions:
                for key in ('pdbUrl', 'cifUrl'):
                    url = pred.get(key)
                    if url and url not in urls:
                        urls.append(url)
    except Exception as e:
        logger.warning(f"AlphaFold API lookup failed for {uniprot_id}: {e}")

    for url_template in AF_URLS:
        url = url_template.format(uniprot_id=uniprot_id)
        if url not in urls:
            urls.append(url)
    return urls


def _local_alphafold_path(uniprot_id, url, pdb_dir):
    """Build a local filename that keeps the AlphaFold model version visible."""
    filename = Path(url.split('?', 1)[0]).name
    if filename.startswith(f"AF-{uniprot_id}-"):
        return os.path.join(pdb_dir, filename)
    suffix = Path(filename).suffix or '.pdb'
    return os.path.join(pdb_dir, f"AF-{uniprot_id}-F1-model_latest{suffix}")


def download_alphafold_structure(uniprot_id, pdb_dir):
    """Download AlphaFold PDB for a UniProt ID. Returns local path or None."""
    os.makedirs(pdb_dir, exist_ok=True)

    existing = []
    for suffix in ("pdb", "cif"):
        existing.extend(Path(pdb_dir).glob(f"AF-{uniprot_id}-F1-model_v*.{suffix}"))
    if existing:
        return str(sorted(existing, reverse=True)[0])

    for url in _alphafold_download_urls(uniprot_id):
        local_path = _local_alphafold_path(uniprot_id, url, pdb_dir)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(resp.content)
            logger.info(f"Downloaded AlphaFold model for {uniprot_id}")
            return local_path
        except requests.exceptions.HTTPError:
            continue
        except Exception:
            continue
    return None


def extract_plddt_scores(structure):
    """Extract pLDDT confidence scores (stored in B-factor field)."""
    try:
        chain = structure[0].child_list[0]
        plddt_scores = [res['CA'].get_bfactor() for res in chain if 'CA' in res]
        if not plddt_scores:
            return {k: np.nan for k in ['avg_plddt', 'min_plddt', 'max_plddt',
                                         'plddt_very_high_pct', 'plddt_high_pct',
                                         'plddt_medium_pct', 'plddt_low_pct']}
        arr = np.array(plddt_scores)
        return {
            'avg_plddt': float(arr.mean()),
            'min_plddt': float(arr.min()),
            'max_plddt': float(arr.max()),
            'plddt_very_high_pct': float((arr > 90).mean()),
            'plddt_high_pct': float(((arr > 70) & (arr <= 90)).mean()),
            'plddt_medium_pct': float(((arr > 50) & (arr <= 70)).mean()),
            'plddt_low_pct': float((arr <= 50).mean()),
        }
    except Exception as e:
        logger.error(f"Error extracting pLDDT: {e}")
        return {k: np.nan for k in ['avg_plddt', 'min_plddt', 'max_plddt',
                                     'plddt_very_high_pct', 'plddt_high_pct',
                                     'plddt_medium_pct', 'plddt_low_pct']}


def verify_structure(structure):
    """Verify structure has necessary components for feature extraction."""
    if not structure:
        return False, "Structure is empty"
    try:
        chain = structure[0].child_list[0]
        ca_atoms = [res['CA'] for res in chain if 'CA' in res]
        if len(ca_atoms) < 10:
            return False, f"Too few CA atoms ({len(ca_atoms)})"
        return True, f"Verified, {len(ca_atoms)} CA atoms"
    except Exception as e:
        return False, f"Verification failed: {e}"


def calculate_contact_order(structure, distance_threshold=8.0, min_separation=4):
    """Compute relative contact order (RCO) from CA atoms."""
    try:
        chain = structure[0].child_list[0]
        ca_atoms = [res['CA'] for res in chain if 'CA' in res]
        L = len(ca_atoms)
        if L < 2:
            return np.nan
        contacts, total_sep = 0, 0.0
        for i in range(L):
            for j in range(i + min_separation, L):
                dist = np.linalg.norm(ca_atoms[i].get_coord() - ca_atoms[j].get_coord())
                if dist <= distance_threshold:
                    contacts += 1
                    total_sep += abs(j - i)
        return (total_sep / contacts) / L if contacts else 0.0
    except Exception as e:
        logger.error(f"Error in calculate_contact_order: {e}")
        return np.nan


def calculate_surface_exposure(structure, percentile=70):
    """Estimate fraction of exposed residues by CA distance from centroid."""
    try:
        chain = structure[0].child_list[0]
        coords = np.array([res['CA'].get_coord() for res in chain if 'CA' in res])
        if coords.size == 0:
            return np.nan
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        threshold = np.percentile(dists, percentile)
        return float((dists > threshold).sum() / len(dists))
    except Exception as e:
        logger.error(f"Error in calculate_surface_exposure: {e}")
        return np.nan


def extract_secondary_structure(structure):
    """Secondary structure detection using geometric criteria on CA distances."""
    try:
        chain = structure[0].child_list[0]
        valid_residues = [res for res in chain if 'CA' in res]
        L = len(valid_residues)
        if L < 5:
            return {'helix_percent': 0.0, 'sheet_percent': 0.0, 'loop_percent': 1.0,
                    'helix_sheet_contrast': -1.0, 'ordered_percent': 0.0}

        ss = ['L'] * L
        ca_coords = np.array([res['CA'].get_coord() for res in valid_residues])
        distances = np.zeros((L, L))
        for i in range(L):
            for j in range(i + 1, L):
                d = np.linalg.norm(ca_coords[i] - ca_coords[j])
                distances[i, j] = distances[j, i] = d

        helix_spans = []
        for i in range(L - 4):
            cons_ok = all(3.6 <= distances[j, j + 1] <= 4.0 for j in range(i, i + 4))
            i3_ok = 4.8 <= distances[i, i + 3] <= 5.8
            i4_ok = 5.8 <= distances[i, i + 4] <= 6.8
            if cons_ok and i3_ok and i4_ok:
                helix_spans.append((i, i + 4))

        sheet_spans = []
        for i in range(L - 2):
            if any(s <= i <= e for s, e in helix_spans):
                continue
            d1_ok = 3.7 <= distances[i, i + 1] <= 4.1 if i + 1 < L else False
            zz_ok = 6.3 <= distances[i, i + 2] <= 7.5 if i + 2 < L else False
            if d1_ok and zz_ok:
                sheet_spans.append((i, i + 2))

        for s, e in helix_spans:
            for i in range(s, min(e + 1, L)):
                ss[i] = 'H'
        for s, e in sheet_spans:
            for i in range(s, min(e + 1, L)):
                if ss[i] == 'L':
                    ss[i] = 'E'

        h, e_cnt = ss.count('H'), ss.count('E')
        hp, sp = h / L, e_cnt / L
        return {'helix_percent': hp, 'sheet_percent': sp, 'loop_percent': 1 - hp - sp,
                'helix_sheet_contrast': hp - sp, 'ordered_percent': hp + sp}
    except Exception as e:
        logger.error(f"Error in extract_secondary_structure: {e}")
        return {k: np.nan for k in ['helix_percent', 'sheet_percent', 'loop_percent',
                                     'helix_sheet_contrast', 'ordered_percent']}


def calculate_avg_cb_distance(structure):
    """Mean distance of CB (or CA for Gly) atoms from centroid."""
    try:
        chain = structure[0].child_list[0]
        coords = []
        for res in chain:
            if res.get_resname() == 'GLY':
                if 'CA' in res:
                    coords.append(res['CA'].get_coord())
            elif 'CB' in res:
                coords.append(res['CB'].get_coord())
            elif 'CA' in res:
                coords.append(res['CA'].get_coord())
        if len(coords) < 3:
            return np.nan
        coords = np.array(coords)
        return float(np.linalg.norm(coords - coords.mean(axis=0), axis=1).mean())
    except Exception as e:
        logger.error(f"Error in calculate_avg_cb_distance: {e}")
        return np.nan


def calculate_compactness(structure):
    """Radius of gyration over CA atoms."""
    try:
        chain = structure[0].child_list[0]
        coords = np.array([res['CA'].get_coord() for res in chain if 'CA' in res])
        if coords.size == 0:
            return np.nan
        centroid = coords.mean(axis=0)
        return float(np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean()))
    except Exception as e:
        logger.error(f"Error in calculate_compactness: {e}")
        return np.nan


def extract_features(entry, pdb_file):
    """Extract all structural features from a single PDB file."""
    if not pdb_file:
        return None
    try:
        struct = parse_structure(pdb_file)
        if struct is None:
            return None
        is_valid, msg = verify_structure(struct)
        if not is_valid:
            logger.warning(f"Validation failed for {entry}: {msg}")
            return None

        feats = {'Entry': entry, 'pdb_path': pdb_file}
        feats.update(extract_plddt_scores(struct))
        feats['rco'] = calculate_contact_order(struct)
        feats['surface_exposure'] = calculate_surface_exposure(struct)
        feats.update(extract_secondary_structure(struct))
        feats['avg_cb_distance'] = calculate_avg_cb_distance(struct)
        feats['compactness'] = calculate_compactness(struct)

        if not np.isnan(feats['compactness']) and feats['compactness'] > 0:
            feats['structural_compactness'] = 1.0 / feats['compactness']
        else:
            feats['structural_compactness'] = np.nan
        if not np.isnan(feats['avg_cb_distance']) and feats['avg_cb_distance'] > 0:
            feats['centralization'] = 1.0 / feats['avg_cb_distance']
        else:
            feats['centralization'] = np.nan

        return feats
    except Exception as e:
        logger.warning(f"Feature extraction failed for {entry}: {e}")
        return None


def main(csv_path, pdb_dir, output_path, skip_download=False):
    """
    Extract structural features for all proteins in a CSV.

    Args:
        csv_path: Input CSV with 'Entry' column.
        pdb_dir: Directory containing (or to download) PDB files.
        output_path: Output CSV path.
        skip_download: If True, only use existing local structures.
    """
    meta_df = pd.read_csv(csv_path)
    if 'Entry' not in meta_df.columns:
        raise ValueError("CSV must contain 'Entry' column")

    entries = meta_df['Entry'].astype(str).tolist()
    logger.info(f"Processing {len(entries)} entries")

    existing = scan_existing_structures(pdb_dir)
    entry_to_file = {}
    missing = []

    for entry in entries:
        if entry in existing:
            entry_to_file[entry] = existing[entry]
        else:
            missing.append(entry)

    logger.info(f"Found {len(entry_to_file)} existing, {len(missing)} missing")

    if missing and not skip_download:
        logger.info(f"Downloading {len(missing)} structures from AlphaFold DB...")
        for entry in tqdm(missing, desc="Downloading"):
            path = download_alphafold_structure(entry, pdb_dir)
            entry_to_file[entry] = path
    else:
        for entry in missing:
            entry_to_file[entry] = None

    records = []
    for entry, pdb_file in tqdm(entry_to_file.items(), desc="Extracting features"):
        if pdb_file is None:
            continue
        feats = extract_features(entry, pdb_file)
        if feats:
            records.append(feats)

    if not records:
        logger.error("No features extracted!")
        return None

    logger.info(f"Extracted features for {len(records)} proteins")
    feat_df = pd.DataFrame(records)

    overlapping = set(meta_df.columns) & set(feat_df.columns) - {'Entry'}
    if overlapping:
        meta_df = meta_df.drop(columns=list(overlapping))

    df = pd.merge(meta_df, feat_df, on='Entry', how='inner')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Structural features written to {output_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Extract structural features from PDB files")
    parser.add_argument("--input", required=True, help="Input CSV with Entry column")
    parser.add_argument("--pdb-dir", required=True, help="Directory for PDB files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--skip-download", action="store_true", help="Only use existing structures")
    args = parser.parse_args()
    main(args.input, args.pdb_dir, args.output, args.skip_download)
