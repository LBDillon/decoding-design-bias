"""Replace UniProt sequence with PDB chain sequence for PDB-mode scoring.

For each protein in expansion_for_scoring_monomer_FINAL.csv with pdb_available=True:
  1. Download the PDB (cached)
  2. Extract the chain sequence from ATOM records
  3. Compute alignment offset and identity vs UniProt sequence
  4. Replace 'sequence' with the chain sequence (so the notebook scores chain vs chain)
  5. Add diagnostic columns: pdb_chain_offset, pdb_chain_identity, uniprot_sequence_length

Filtering:
  - Drop rows where pdb_available is False
  - Drop rows where best-alignment identity < IDENTITY_THRESHOLD
  - Drop rows where the chain has any non-standard residues that map to 'X'
    (these will fail in ProteinMPNN scoring)

Output: dataset_update/expansion_for_scoring_monomer_PDB_FINAL.csv
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_FINAL.csv'
OUTPUT = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_PDB_FINAL.csv'
PDB_CACHE = REPO_ROOT / 'dataset_update' / 'pdb_cache'

IDENTITY_THRESHOLD = 0.95  # drop proteins where chain sequence diverges from UniProt
ALLOW_X_IN_CHAIN = False    # drop chains containing 'X' (non-standard residues)

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
MODIFIED_TO_PARENT = {
    'MSE': 'M', 'SEP': 'S', 'TPO': 'T', 'PTR': 'Y', 'HYP': 'P',
    'PCA': 'E', 'CME': 'C', 'CSO': 'C', 'KCX': 'K', 'MLY': 'K',
    'LLP': 'K', 'CSD': 'C', 'OCS': 'C', 'CAS': 'C',
}


def download_pdb(pdb_id: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 100:
        return True
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            dest.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def parse_chain_sequence(pdb_path: Path, chain: str) -> str:
    """Extract a chain's sequence from ATOM/HETATM CA records."""
    seen = {}  # (resnum, icode) -> 3-letter code
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            res_name = line[17:20].strip()
            chain_id = line[21]
            if chain_id != chain:
                continue
            try:
                res_num = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            key = (res_num, icode)
            if key not in seen:
                seen[key] = res_name

    chain_residues = sorted(seen.items(), key=lambda x: x[0])
    seq = []
    for (rn, ic), code in chain_residues:
        if code in THREE_TO_ONE:
            seq.append(THREE_TO_ONE[code])
        elif code in MODIFIED_TO_PARENT:
            seq.append(MODIFIED_TO_PARENT[code])
        else:
            seq.append('X')
    return ''.join(seq)


def best_alignment(uniprot_seq: str, chain_seq: str) -> tuple[int, float]:
    """Find the offset of chain_seq within uniprot_seq with highest identity.

    Returns (best_offset, best_identity).
    """
    n_chain = len(chain_seq)
    n_uni = len(uniprot_seq)
    if n_chain == 0:
        return 0, 0.0
    if n_chain > n_uni:
        # Chain longer than UniProt - try aligning UniProt within chain
        best_offset, best_id = 0, 0.0
        for offset in range(n_chain - n_uni + 1):
            window = chain_seq[offset:offset + n_uni]
            ident = sum(1 for x, y in zip(window, uniprot_seq) if x == y) / n_uni
            if ident > best_id:
                best_id = ident
                best_offset = -offset  # negative means UniProt starts inside chain
        return best_offset, best_id

    best_offset, best_id = 0, 0.0
    for offset in range(n_uni - n_chain + 1):
        window = uniprot_seq[offset:offset + n_chain]
        ident = sum(1 for x, y in zip(window, chain_seq) if x == y) / n_chain
        if ident > best_id:
            best_id = ident
            best_offset = offset
    return best_offset, best_id


def process_one(row: pd.Series) -> dict:
    """Worker: returns dict with chain sequence + alignment info."""
    pdb_id = row['pdb_id']
    chain = row['pdb_chain']
    uniprot_seq = row['sequence']
    pdb_path = PDB_CACHE / f"{pdb_id}.pdb"

    if not download_pdb(pdb_id, pdb_path):
        return {'Entry': row['Entry'], 'pdb_chain_sequence': None,
                'pdb_chain_offset': None, 'pdb_chain_identity': 0.0,
                'len_chain': 0, 'fail_reason': 'download_failed'}

    chain_seq = parse_chain_sequence(pdb_path, chain)
    if not chain_seq:
        return {'Entry': row['Entry'], 'pdb_chain_sequence': None,
                'pdb_chain_offset': None, 'pdb_chain_identity': 0.0,
                'len_chain': 0, 'fail_reason': 'chain_not_found'}

    offset, identity = best_alignment(uniprot_seq, chain_seq)
    has_x = 'X' in chain_seq

    return {
        'Entry': row['Entry'],
        'pdb_chain_sequence': chain_seq,
        'pdb_chain_offset': offset,
        'pdb_chain_identity': round(identity, 4),
        'len_chain': len(chain_seq),
        'has_x_in_chain': has_x,
        'fail_reason': None,
    }


def main():
    PDB_CACHE.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"Loaded {len(df):,} entries from {INPUT.name}")

    # Restrict to entries with PDB available
    pdb_df = df[df['pdb_available']].reset_index(drop=True)
    print(f"  with pdb_available=True: {len(pdb_df):,}")

    # Parallel processing (most time is download - Disk-bound on cache hits)
    results = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(process_one, row): i for i, row in pdb_df.iterrows()}
        for f in tqdm(as_completed(futures), total=len(futures),
                       desc='Extracting chain seqs'):
            results.append(f.result())

    info = pd.DataFrame(results)
    merged = pdb_df.merge(info, on='Entry', how='left')

    # Stats before filtering
    print(f"\n=== Pre-filter stats (n={len(merged):,}) ===")
    print(f"  Identity == 1.000:        {(merged['pdb_chain_identity'] >= 0.999).sum():>5,d}")
    print(f"  Identity ≥ 0.95:          {(merged['pdb_chain_identity'] >= 0.95).sum():>5,d}")
    print(f"  Identity ≥ 0.90:          {(merged['pdb_chain_identity'] >= 0.90).sum():>5,d}")
    print(f"  Identity < 0.90:          {(merged['pdb_chain_identity'] < 0.90).sum():>5,d}")
    print(f"  Offset == 0 (no shift):   {(merged['pdb_chain_offset'] == 0).sum():>5,d}")
    print(f"  Offset ≠ 0 (shifted):     {(merged['pdb_chain_offset'] != 0).sum():>5,d}")
    print(f"  Has 'X' in chain:         {merged['has_x_in_chain'].sum():>5,d}")
    print(f"  Download/chain failures:  {merged['fail_reason'].notna().sum():>5,d}")

    print(f"\n  Offset distribution (top 10):")
    print(merged['pdb_chain_offset'].value_counts().head(10).to_string())

    # Apply filters
    print(f"\n=== Filtering ===")
    keep = (
        merged['pdb_chain_identity'] >= IDENTITY_THRESHOLD
    )
    if not ALLOW_X_IN_CHAIN:
        keep = keep & (~merged['has_x_in_chain'].fillna(True))
    keep = keep & merged['fail_reason'].isna()

    final = merged[keep].copy()

    # Replace 'sequence' column with the chain sequence (the notebook reads 'sequence')
    final['uniprot_sequence_length'] = final['sequence'].str.len()
    final['sequence'] = final['pdb_chain_sequence']

    # Drop helper cols not needed for scoring
    final = final.drop(columns=['has_x_in_chain', 'fail_reason'])

    print(f"  Kept: {len(final):,} / {len(merged):,} ({100*len(final)/len(merged):.1f}%)")

    # Domain breakdown
    print(f"\n=== Final dataset domain distribution ===")
    print(final['domain'].value_counts().to_string())

    final.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(final):,} entries to {OUTPUT.name}")
    print(f"  Path: {OUTPUT}")


if __name__ == '__main__':
    main()
