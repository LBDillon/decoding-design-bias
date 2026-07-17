"""decoding_bias.dataset.expansion -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - fetch_and_annotate_round2
  - fetch_and_annotate_round3
  - build_round3_family_targets
  - inject_round2_scores
"""

import argparse
import json
import numpy as np
import pandas as pd
import re
import requests
import sys
import time
from pathlib import Path
from tqdm import tqdm
from decoding_bias.features.sequence_features import calculate_sequence_features
from decoding_bias.features.structural_features import download_alphafold_structure, extract_features, scan_existing_structures
from dataset_update.protein_classification import classify
from io import StringIO

# ---------- from fetch_and_annotate_round2.py ----------
fetch_and_annotate_round2_HERE = Path(__file__).parent
fetch_and_annotate_round2_ROOT = fetch_and_annotate_round2_HERE.parent
ROUND2 = fetch_and_annotate_round2_HERE / 'round2'
fetch_and_annotate_round2_PDB_CACHE = fetch_and_annotate_round2_HERE / 'alphafold_cache'
fetch_and_annotate_round2_UNIPROT_SEARCH = 'https://rest.uniprot.org/uniprotkb/search'
fetch_and_annotate_round2_DOMAIN_TAXID = {'Archaea': 2157, 'Bacteria': 2, 'Eukaryota': 2759}
FUNCTION_QUERY = {'transferase': 'ec:2*', 'hydrolase': 'ec:3*', 'oxidoreductase': 'ec:1*', 'lyase': 'ec:4*', 'isomerase': 'ec:5*', 'ligase': 'ec:6*', 'translocase': 'ec:7*', 'transcription': 'keyword:KW-0804', 'membrane': 'keyword:KW-0472', 'GTPase': 'keyword:KW-0342', 'electron_carrier': 'keyword:KW-0249', 'chaperone': 'keyword:KW-0143', 'signaling': 'keyword:KW-0807', 'transport': 'keyword:KW-0813', 'RNA-binding': 'keyword:KW-0694', 'DNA-binding': 'keyword:KW-0238', 'cytoskeletal': 'keyword:KW-0206', 'structural': 'keyword:KW-0729', 'protease_inhibitor': 'keyword:KW-0646'}
fetch_and_annotate_round2_BASE_FILTER = 'reviewed:true AND length:[50 TO 1000] AND NOT keyword:KW-0689'
fetch_and_annotate_round2_OVERSAMPLE_FACTOR = 3
PER_CELL_MAX = 1500
fetch_and_annotate_round2_MIN_PLDDT = 70.0
fetch_and_annotate_round2_MIN_LEN = 50
fetch_and_annotate_round2_MAX_LEN = 1000
fetch_and_annotate_round2_FIELDS = ','.join(['accession', 'id', 'protein_name', 'gene_names', 'organism_name', 'organism_id', 'lineage', 'length', 'sequence', 'ec', 'keyword', 'cc_function', 'cc_subcellular_location', 'cc_subunit', 'xref_pfam', 'protein_families'])
def fetch_and_annotate_round2_stage_queries(dry_run=False):
    gap = pd.read_csv(fetch_and_annotate_round2_HERE / 'expansion_round2_gap_target.csv')
    queries = []
    for (_, row) in gap.iterrows():
        (domain, func, add) = (row['domain'], row['broad_function'], int(row['add_target']))
        if domain not in fetch_and_annotate_round2_DOMAIN_TAXID or func not in FUNCTION_QUERY:
            continue
        taxid = fetch_and_annotate_round2_DOMAIN_TAXID[domain]
        func_q = FUNCTION_QUERY[func]
        query = f'({fetch_and_annotate_round2_BASE_FILTER}) AND (taxonomy_id:{taxid}) AND ({func_q})'
        size = min(add * fetch_and_annotate_round2_OVERSAMPLE_FACTOR, PER_CELL_MAX)
        queries.append({'cell_id': f'{domain}__{func}', 'domain': domain, 'broad_function': func, 'target_n': add, 'fetch_n': size, 'query': query})
    out = ROUND2 / 'queries.json'
    out.write_text(json.dumps(queries, indent=2))
    print(f"Built {len(queries)} queries, total fetch={sum((q['fetch_n'] for q in queries))}, target={sum((q['target_n'] for q in queries))}")
    print(f'Wrote {out}')
    if dry_run:
        print('\nFirst 3 queries:')
        for q in queries[:3]:
            print(f"  [{q['cell_id']}] target={q['target_n']} fetch={q['fetch_n']}")
            print(f"    {q['query']}")
    return queries
def fetch_and_annotate_round2__uniprot_paged_search(query, want_n, fields=fetch_and_annotate_round2_FIELDS):
    """Fetch up to want_n Swiss-Prot entries for a query, with pagination."""
    rows = []
    fetched = 0
    cursor = None
    while fetched < want_n:
        page_size = min(500, want_n - fetched)
        params = {'query': query, 'format': 'tsv', 'fields': fields, 'size': page_size}
        if cursor:
            params['cursor'] = cursor
        r = requests.get(fetch_and_annotate_round2_UNIPROT_SEARCH, params=params, timeout=30)
        if r.status_code != 200:
            print(f'  UniProt {r.status_code}: {r.text[:200]}')
            break
        from io import StringIO
        page = pd.read_csv(StringIO(r.text), sep='\t')
        if len(page) == 0:
            break
        rows.append(page)
        fetched += len(page)
        link = r.headers.get('Link', '')
        cursor = None
        if 'rel="next"' in link:
            import re
            m = re.search('cursor=([^&>]+)', link)
            if m:
                cursor = m.group(1)
        if cursor is None:
            break
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).head(want_n)
def fetch_and_annotate_round2_stage_fetch(dry_run=False):
    queries = json.loads((ROUND2 / 'queries.json').read_text())
    out_path = ROUND2 / 'candidates_raw.csv'
    if dry_run:
        print(f"Would fetch {sum((q['fetch_n'] for q in queries))} entries across {len(queries)} cells")
        return
    all_rows = []
    for q in tqdm(queries, desc='UniProt cells'):
        page = fetch_and_annotate_round2__uniprot_paged_search(q['query'], q['fetch_n'])
        if len(page):
            page['cell_id'] = q['cell_id']
            page['target_domain'] = q['domain']
            page['target_broad_function'] = q['broad_function']
            all_rows.append(page)
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(out_path, index=False)
        print(f'Fetched {len(out)} candidate rows, wrote {out_path}')
    else:
        print('No results returned from UniProt.')
def _normalize_species(name):
    import re
    if pd.isna(name):
        return name
    name = re.sub('\\s*\\(strain[^)]*\\)', '', name)
    name = re.sub('\\s*\\([^)]+\\)', '', name).strip()
    return name
def fetch_and_annotate_round2_stage_filter():
    raw = pd.read_csv(ROUND2 / 'candidates_raw.csv')
    merged = pd.read_csv(fetch_and_annotate_round2_HERE / 'merged_dataset.csv')
    existing = set(merged['Entry'])
    rename_map = {'Entry': 'Entry', 'Accession': 'Entry', 'Entry Name': 'EntryName', 'Protein names': 'protein_name', 'Gene Names': 'gene_names', 'Organism': 'Organism', 'Organism (ID)': 'organism_id', 'Lineage': 'lineage', 'Taxonomic lineage': 'lineage', 'Taxonomic lineage (Ids)': 'lineage_ids', 'Length': 'Length', 'Sequence': 'sequence', 'EC number': 'ec', 'Keywords': 'keywords', 'Function [CC]': 'cc_function', 'Subcellular location [CC]': 'cc_subcellular_location', 'Subunit structure [CC]': 'cc_subunit', 'Pfam': 'Pfam', 'Protein families': 'protein_families_raw'}
    raw = raw.rename(columns={k: v for (k, v) in rename_map.items() if k in raw.columns})
    before = len(raw)
    raw = raw.drop_duplicates(subset=['Entry'], keep='first')
    print(f'After dedup: {len(raw)} (was {before})')
    raw = raw[(raw['Length'] >= fetch_and_annotate_round2_MIN_LEN) & (raw['Length'] <= fetch_and_annotate_round2_MAX_LEN)]
    print(f'After length filter: {len(raw)}')
    raw = raw[~raw['Entry'].isin(existing)]
    print(f'After existing-dataset exclusion: {len(raw)}')

    def _ok_seq(s):
        if not isinstance(s, str) or len(s) == 0:
            return False
        bad = set(s.upper()) - set('ACDEFGHIKLMNPQRSTVWY')
        return len(bad) == 0 or bad == {'U'} or bad == {'X'} or (bad <= {'U', 'X'})
    raw = raw[raw['sequence'].apply(_ok_seq)]
    print(f'After sequence-validity filter: {len(raw)}')
    raw['species'] = raw['Organism'].apply(_normalize_species)

    def _domain(line):
        if not isinstance(line, str):
            return None
        for d in ('Viruses', 'Archaea', 'Bacteria', 'Eukaryota'):
            if d in line:
                return d
        return None
    raw['domain'] = raw['lineage'].apply(_domain) if 'lineage' in raw.columns else None
    raw.to_csv(ROUND2 / 'candidates_filtered.csv', index=False)
    print(f"Wrote {ROUND2 / 'candidates_filtered.csv'}: {len(raw)} candidates")
def fetch_and_annotate_round2_stage_structures(limit=None):
    df = pd.read_csv(ROUND2 / 'candidates_filtered.csv')
    if limit:
        df = df.head(limit)
    existing = scan_existing_structures(str(fetch_and_annotate_round2_PDB_CACHE))
    status = []
    for (_, row) in tqdm(df.iterrows(), total=len(df), desc='AF download'):
        entry = row['Entry']
        path = existing.get(entry)
        if path and Path(path).exists():
            status.append({'Entry': entry, 'ok': True, 'path': str(path), 'cached': True})
            continue
        try:
            result = download_alphafold_structure(entry, str(fetch_and_annotate_round2_PDB_CACHE))
            ok = result is not None and Path(result).exists()
            status.append({'Entry': entry, 'ok': ok, 'path': str(result) if ok else '', 'cached': False})
        except Exception as e:
            status.append({'Entry': entry, 'ok': False, 'path': '', 'error': str(e), 'cached': False})
        time.sleep(0.05)
    s = pd.DataFrame(status)
    s.to_csv(ROUND2 / 'structure_status.csv', index=False)
    print(f"AF success: {s['ok'].sum()}/{len(s)}")
def fetch_and_annotate_round2_stage_features():
    df = pd.read_csv(ROUND2 / 'candidates_filtered.csv')
    status = pd.read_csv(ROUND2 / 'structure_status.csv')
    ok_status = status[status['ok']].copy()
    path_by_entry = dict(zip(ok_status['Entry'], ok_status['path']))
    have_struct = set(path_by_entry)
    df = df[df['Entry'].isin(have_struct)].copy()
    print(f'Computing features for {len(df)} entries with AF structures')
    seq_feats = []
    for (_, row) in tqdm(df.iterrows(), total=len(df), desc='Seq features'):
        f = calculate_sequence_features(str(row['sequence']))
        f['Entry'] = row['Entry']
        seq_feats.append(f)
    seq_df = pd.DataFrame(seq_feats)
    struct_feats = []
    for (_, row) in tqdm(df.iterrows(), total=len(df), desc='Struct features'):
        path = path_by_entry.get(row['Entry'])
        try:
            f = extract_features(row['Entry'], str(path))
            if f is None:
                f = {'Entry': row['Entry'], '_err': 'feature extraction failed'}
            struct_feats.append(f)
        except Exception as e:
            struct_feats.append({'Entry': row['Entry'], '_err': str(e)})
    struct_df = pd.DataFrame(struct_feats)
    out = df.merge(seq_df, on='Entry', how='left', suffixes=('', '_seq'))
    out = out.merge(struct_df, on='Entry', how='left', suffixes=('', '_struct'))
    if 'avg_plddt' in out.columns:
        before = len(out)
        out = out[out['avg_plddt'].fillna(0) >= fetch_and_annotate_round2_MIN_PLDDT]
        print(f'pLDDT≥{fetch_and_annotate_round2_MIN_PLDDT} filter: kept {len(out)}/{before}')
    out.to_csv(ROUND2 / 'features.csv', index=False)
    print(f"Wrote {ROUND2 / 'features.csv'}: {len(out)} entries")
def fetch_and_annotate_round2_stage_classify():
    df = pd.read_csv(ROUND2 / 'features.csv')
    rename = {'Pfam': 'Pfam', 'ec': 'EC number', 'protein_families_raw': 'Protein families', 'Organism': 'Organism', 'protein_name': 'Protein names'}
    work = df.rename(columns={k: v for (k, v) in rename.items() if k in df.columns})
    classified = classify(work)
    out = df.copy()
    for col in ('protein_family', 'broad_function', 'protein_name_clean', 'is_enzyme', 'is_transmembrane', 'is_glycosylated', 'has_disordered'):
        if col in classified.columns:
            out[col] = classified[col].values
    out.to_csv(ROUND2 / 'classified.csv', index=False)
    print(f"Classified {len(out)} entries; wrote {ROUND2 / 'classified.csv'}")
def fetch_and_annotate_round2_stage_finalize():
    df = pd.read_csv(ROUND2 / 'classified.csv')
    gap = pd.read_csv(fetch_and_annotate_round2_HERE / 'expansion_round2_gap_target.csv')
    targets = {(r['domain'], r['broad_function']): int(r['add_target']) for (_, r) in gap.iterrows()}
    drop_bf = {'ribosomal', 'translation_factor'}
    before = len(df)
    df = df[~df['broad_function'].isin(drop_bf)]
    print(f'Dropped {before - len(df)} entries reclassified as ribosomal/translation_factor')
    keep = []
    for ((d, f), tgt) in targets.items():
        sub = df[(df['target_domain'] == d) & (df['target_broad_function'] == f)].copy()
        if sub.empty:
            print(f'  [{d}/{f}] 0 candidates, target {tgt} - UNMET')
            continue
        per_sp_cap = max(2, tgt // 20)
        sub = sub.groupby('species', group_keys=False).head(per_sp_cap)
        sub = sub.head(tgt)
        keep.append(sub)
        print(f'  [{d}/{f}] kept {len(sub)}/{tgt}')
    if not keep:
        print('Nothing to write.')
        return
    final = pd.concat(keep, ignore_index=True).drop_duplicates('Entry')
    before = len(final)
    final = final[final['target_domain'] != 'Viruses']
    print(f'Dropped {before - len(final)} viral entries (AFDB coverage is too sparse)')
    final['domain'] = final['target_domain']
    final['source'] = 'expansion_round2'
    final['structure_source'] = 'AF'
    out = ROUND2 / 'expansion_round2_for_scoring.csv'
    final.to_csv(out, index=False)
    print(f'\nFinal round-2 expansion: {len(final)} proteins')
    print(f"  by domain: {final['domain'].value_counts().to_dict()}")
    print(f"  by broad_function (top 10): {final['broad_function'].value_counts().head(10).to_dict()}")
    print(f"  unique species: {final['species'].nunique()}")
    print(f"  ribosomal proteins (should be 0): {(final['broad_function'] == 'ribosomal').sum()}")
    print(f'Wrote {out}')
fetch_and_annotate_round2_STAGES = {'queries': fetch_and_annotate_round2_stage_queries, 'fetch': fetch_and_annotate_round2_stage_fetch, 'filter': fetch_and_annotate_round2_stage_filter, 'structures': fetch_and_annotate_round2_stage_structures, 'features': fetch_and_annotate_round2_stage_features, 'classify': fetch_and_annotate_round2_stage_classify, 'finalize': fetch_and_annotate_round2_stage_finalize}
def fetch_and_annotate_round2_main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=list(fetch_and_annotate_round2_STAGES) + ['all'], help="Which stage to run (or 'all')")
    ap.add_argument('--dry-run', action='store_true', help='Preview without making network calls (where supported)')
    ap.add_argument('--limit', type=int, default=None, help='Cap entries processed (debugging)')
    args = ap.parse_args()
    if args.stage == 'all':
        for name in ['queries', 'fetch', 'filter', 'structures', 'features', 'classify', 'finalize']:
            print(f'\n===== Stage: {name} =====')
            fn = fetch_and_annotate_round2_STAGES[name]
            kwargs = {}
            if name == 'queries' or name == 'fetch':
                kwargs['dry_run'] = args.dry_run
            if name == 'structures' and args.limit is not None:
                kwargs['limit'] = args.limit
            fn(**kwargs)
    else:
        fn = fetch_and_annotate_round2_STAGES[args.stage]
        kwargs = {}
        if args.stage in ('queries', 'fetch'):
            kwargs['dry_run'] = args.dry_run
        if args.stage == 'structures' and args.limit is not None:
            kwargs['limit'] = args.limit
        fn(**kwargs)
def fetch_and_annotate_round2__entry():
    sys.path.insert(0, str(fetch_and_annotate_round2_ROOT))
    ROUND2.mkdir(exist_ok=True)
    fetch_and_annotate_round2_PDB_CACHE.mkdir(exist_ok=True)
    fetch_and_annotate_round2_main()

# ---------- from fetch_and_annotate_round3.py ----------
fetch_and_annotate_round3_HERE = Path(__file__).parent
fetch_and_annotate_round3_ROOT = fetch_and_annotate_round3_HERE.parent
ROUND3 = fetch_and_annotate_round3_HERE / 'round3'
fetch_and_annotate_round3_PDB_CACHE = fetch_and_annotate_round3_HERE / 'alphafold_cache'
fetch_and_annotate_round3_UNIPROT_SEARCH = 'https://rest.uniprot.org/uniprotkb/search'
TARGETS_CSV = fetch_and_annotate_round3_HERE / 'expansion_round3_family_targets.csv'
fetch_and_annotate_round3_OVERSAMPLE_FACTOR = 3
PER_CELL_MAX_FETCH = 200
PER_SPECIES_CAP = 3
fetch_and_annotate_round3_MIN_PLDDT = 70.0
fetch_and_annotate_round3_MIN_LEN = 50
fetch_and_annotate_round3_MAX_LEN = 1000
fetch_and_annotate_round3_FIELDS = ','.join(['accession', 'id', 'protein_name', 'gene_names', 'organism_name', 'organism_id', 'lineage', 'length', 'sequence', 'ec', 'keyword', 'cc_function', 'cc_subcellular_location', 'cc_subunit', 'xref_pfam', 'protein_families'])
def _norm_species(name):
    if pd.isna(name):
        return name
    name = re.sub('\\s*\\(strain[^)]*\\)', '', str(name))
    name = re.sub('\\s*\\([^)]+\\)', '', name).strip()
    return name
def fetch_and_annotate_round3_stage_queries(dry_run=False):
    gap = pd.read_csv(TARGETS_CSV)
    queries = []
    for (_, r) in gap.iterrows():
        target_n = int(r['target_n'])
        if target_n <= 0:
            continue
        fetch_n = min(target_n * fetch_and_annotate_round3_OVERSAMPLE_FACTOR, PER_CELL_MAX_FETCH)
        cell_id = f"{r['protein_family']}__{r['move']}__{r['target_domain']}"
        queries.append({'cell_id': cell_id, 'protein_family': r['protein_family'], 'target_domain': r['target_domain'], 'move': r['move'], 'target_n': target_n, 'fetch_n': fetch_n, 'query': r['query']})
    (ROUND3 / 'queries.json').write_text(json.dumps(queries, indent=2))
    print(f"Built {len(queries)} queries; target={sum((q['target_n'] for q in queries))}; fetch={sum((q['fetch_n'] for q in queries))}")
    if dry_run:
        print('\nSample queries:')
        for q in queries[:5]:
            print(f"  [{q['move']}] {q['protein_family'][:60]} → {q['target_domain']} (target={q['target_n']})")
            print(f"    {q['query']}")
def fetch_and_annotate_round3__uniprot_paged_search(query, want_n, fields=fetch_and_annotate_round3_FIELDS):
    (rows, fetched, cursor) = ([], 0, None)
    while fetched < want_n:
        params = {'query': query, 'format': 'tsv', 'fields': fields, 'size': min(500, want_n - fetched)}
        if cursor:
            params['cursor'] = cursor
        try:
            r = requests.get(fetch_and_annotate_round3_UNIPROT_SEARCH, params=params, timeout=30)
        except requests.RequestException as e:
            print(f'  network error: {e}', flush=True)
            break
        if r.status_code != 200:
            print(f'  UniProt {r.status_code}: {r.text[:200]}', flush=True)
            break
        page = pd.read_csv(StringIO(r.text), sep='\t')
        if len(page) == 0:
            break
        rows.append(page)
        fetched += len(page)
        link = r.headers.get('Link', '')
        cursor = None
        if 'rel="next"' in link:
            m = re.search('cursor=([^&>]+)', link)
            if m:
                cursor = m.group(1)
        if cursor is None:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).head(want_n)
def fetch_and_annotate_round3_stage_fetch(dry_run=False, limit=None):
    queries = json.loads((ROUND3 / 'queries.json').read_text())
    if limit:
        queries = queries[:limit]
    if dry_run:
        print(f"Would fetch {sum((q['fetch_n'] for q in queries))} entries across {len(queries)} cells")
        return
    (all_rows, yield_log) = ([], [])
    for q in tqdm(queries, desc='UniProt cells'):
        page = fetch_and_annotate_round3__uniprot_paged_search(q['query'], q['fetch_n'])
        yield_log.append({'cell_id': q['cell_id'], 'target_n': q['target_n'], 'fetch_n': q['fetch_n'], 'got': len(page)})
        if len(page):
            page['cell_id'] = q['cell_id']
            page['target_protein_family'] = q['protein_family']
            page['target_domain'] = q['target_domain']
            page['target_n'] = q['target_n']
            page['move'] = q['move']
            all_rows.append(page)
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(ROUND3 / 'candidates_raw.csv', index=False)
        pd.DataFrame(yield_log).to_csv(ROUND3 / 'fetch_yield_log.csv', index=False)
        print(f"Fetched {len(out)} candidate rows (with overlap); unique Entries: {(out['Entry'].nunique() if 'Entry' in out.columns else 0)}")
    else:
        print('No results returned from UniProt.')
def fetch_and_annotate_round3_stage_filter():
    raw = pd.read_csv(ROUND3 / 'candidates_raw.csv')
    rename = {'Entry': 'Entry', 'Entry Name': 'EntryName', 'Protein names': 'protein_name', 'Gene Names': 'gene_names', 'Organism': 'Organism', 'Organism (ID)': 'organism_id', 'Lineage': 'lineage', 'Taxonomic lineage': 'lineage', 'Length': 'Length', 'Sequence': 'sequence', 'EC number': 'ec', 'Keywords': 'keywords', 'Function [CC]': 'cc_function', 'Subcellular location [CC]': 'cc_subcellular_location', 'Subunit structure [CC]': 'cc_subunit', 'Pfam': 'Pfam', 'Protein families': 'protein_families_raw'}
    raw = raw.rename(columns={k: v for (k, v) in rename.items() if k in raw.columns})
    pool = set()
    for (path, label) in [(fetch_and_annotate_round3_HERE / 'Decoding_Bias_Dataset_updated.csv', 'main'), (fetch_and_annotate_round3_HERE / 'merged_dataset.csv', 'merged (main+round1)'), (fetch_and_annotate_round3_HERE / 'round2' / 'expansion_round2_KEPT.csv', 'round2 KEPT'), (fetch_and_annotate_round3_HERE / 'round2' / 'expansion_round2_DROPPED.csv', 'round2 DROPPED')]:
        if path.exists():
            ents = set(pd.read_csv(path, usecols=['Entry'])['Entry'])
            pool |= ents
            print(f'  Exclusion pool from {label}: +{len(ents)} (total {len(pool)})')
    print(f'Final exclusion pool: {len(pool)} unique Entries')
    before = len(raw)
    raw = raw.drop_duplicates(subset=['Entry'], keep='first')
    print(f'After dedup within fetch: {len(raw)} (was {before})')
    raw = raw[(raw['Length'] >= fetch_and_annotate_round3_MIN_LEN) & (raw['Length'] <= fetch_and_annotate_round3_MAX_LEN)]
    print(f'After length filter: {len(raw)}')
    raw = raw[~raw['Entry'].isin(pool)]
    print(f'After exclusion-pool filter: {len(raw)}')

    def _ok_seq(s):
        if not isinstance(s, str) or not s:
            return False
        bad = set(s.upper()) - set('ACDEFGHIKLMNPQRSTVWY')
        return len(bad) == 0 or bad <= {'U', 'X'}
    raw = raw[raw['sequence'].apply(_ok_seq)]
    print(f'After sequence-validity filter: {len(raw)}')
    raw['species'] = raw['Organism'].apply(_norm_species)

    def _domain(line):
        if not isinstance(line, str):
            return None
        for d in ('Viruses', 'Archaea', 'Bacteria', 'Eukaryota'):
            if d in line:
                return d
        return None
    raw['domain'] = raw['lineage'].apply(_domain) if 'lineage' in raw.columns else None
    if 'target_domain' in raw.columns:
        raw['domain'] = raw['domain'].fillna(raw['target_domain'])
    raw.to_csv(ROUND3 / 'candidates_filtered.csv', index=False)
    print(f"Wrote {ROUND3 / 'candidates_filtered.csv'}: {len(raw)} candidates")
def fetch_and_annotate_round3_stage_structures(limit=None):
    df = pd.read_csv(ROUND3 / 'candidates_filtered.csv')
    if limit:
        df = df.head(limit)
    status = []
    for (_, row) in tqdm(df.iterrows(), total=len(df), desc='AF download'):
        entry = row['Entry']
        existing = list(fetch_and_annotate_round3_PDB_CACHE.glob(f'AF-{entry}-F1-model_v*.pdb'))
        if existing:
            status.append({'Entry': entry, 'ok': True, 'path': str(existing[0]), 'cached': True})
            continue
        try:
            result = download_alphafold_structure(entry, str(fetch_and_annotate_round3_PDB_CACHE))
            ok = result is not None and Path(result).exists()
            status.append({'Entry': entry, 'ok': ok, 'path': str(result) if ok else '', 'cached': False})
        except Exception as e:
            status.append({'Entry': entry, 'ok': False, 'path': '', 'error': str(e), 'cached': False})
        time.sleep(0.03)
    s = pd.DataFrame(status)
    s.to_csv(ROUND3 / 'structure_status.csv', index=False)
    print(f"AF success: {s['ok'].sum()}/{len(s)}")
def fetch_and_annotate_round3_stage_features():
    df = pd.read_csv(ROUND3 / 'candidates_filtered.csv')
    status = pd.read_csv(ROUND3 / 'structure_status.csv')
    have = set(status.loc[status['ok'], 'Entry'])
    df = df[df['Entry'].isin(have)].copy()
    print(f'Computing features for {len(df)} entries')
    seq_feats = []
    for (_, row) in tqdm(df.iterrows(), total=len(df), desc='seq features'):
        f = calculate_sequence_features(str(row['sequence']))
        f['Entry'] = row['Entry']
        seq_feats.append(f)
    seq_df = pd.DataFrame(seq_feats)
    struct_feats = []
    for (_, row) in tqdm(df.iterrows(), total=len(df), desc='struct features'):
        existing = list(fetch_and_annotate_round3_PDB_CACHE.glob(f"AF-{row['Entry']}-F1-model_v*.pdb"))
        path = existing[0] if existing else None
        if path is None:
            struct_feats.append({'Entry': row['Entry'], '_err': 'no_structure'})
            continue
        try:
            f = extract_features(row['Entry'], str(path))
            struct_feats.append(f)
        except Exception as e:
            struct_feats.append({'Entry': row['Entry'], '_err': str(e)})
    struct_df = pd.DataFrame(struct_feats)
    out = df.merge(seq_df, on='Entry', how='left', suffixes=('', '_seq'))
    out = out.merge(struct_df, on='Entry', how='left', suffixes=('', '_struct'))
    if 'avg_plddt' in out.columns:
        before = len(out)
        out = out[out['avg_plddt'].fillna(0) >= fetch_and_annotate_round3_MIN_PLDDT]
        print(f'pLDDT≥{fetch_and_annotate_round3_MIN_PLDDT} filter: kept {len(out)}/{before}')
    out.to_csv(ROUND3 / 'features.csv', index=False)
    print(f"Wrote {ROUND3 / 'features.csv'}: {len(out)} entries")
def fetch_and_annotate_round3_stage_classify():
    df = pd.read_csv(ROUND3 / 'features.csv')
    rename = {'Pfam': 'Pfam', 'ec': 'EC number', 'protein_families_raw': 'Protein families', 'Organism': 'Organism', 'protein_name': 'Protein names'}
    work = df.rename(columns={k: v for (k, v) in rename.items() if k in df.columns})
    classified = classify(work)
    out = df.copy()
    for col in ('protein_family', 'broad_function', 'protein_name_clean', 'is_enzyme', 'is_transmembrane', 'is_glycosylated', 'has_disordered'):
        if col in classified.columns:
            out[col] = classified[col].values
    out.to_csv(ROUND3 / 'classified.csv', index=False)
    print(f"Classified {len(out)} entries; wrote {ROUND3 / 'classified.csv'}")
def fetch_and_annotate_round3_stage_finalize():
    df = pd.read_csv(ROUND3 / 'classified.csv')
    targets = pd.read_csv(TARGETS_CSV)
    before = len(df)
    df = df[~df['broad_function'].isin({'ribosomal', 'translation_factor'})]
    print(f'Defensive drop of ribosomal/translation_factor: {before - len(df)}')
    keep_rows = []
    cell_report = []
    for (_, t) in targets.iterrows():
        cell_id = f"{t['protein_family']}__{t['move']}__{t['target_domain']}"
        sub = df[df['cell_id'] == cell_id].copy()
        if sub.empty:
            cell_report.append({**t.to_dict(), 'kept': 0, 'status': 'empty'})
            continue
        sub = sub.groupby('species', group_keys=False).head(PER_SPECIES_CAP)
        sub = sub.head(int(t['target_n']))
        cell_report.append({**t.to_dict(), 'kept': len(sub), 'status': 'ok'})
        keep_rows.append(sub)
    if not keep_rows:
        print('Nothing to write.')
        return
    final = pd.concat(keep_rows, ignore_index=True).drop_duplicates('Entry')
    final['source'] = 'expansion_round3'
    final['structure_source'] = 'AF'
    if 'target_domain' in final.columns:
        final['domain'] = final['domain'].fillna(final['target_domain'])
    out = ROUND3 / 'expansion_round3_for_scoring.csv'
    final.to_csv(out, index=False)
    report = pd.DataFrame(cell_report)
    report.to_csv(ROUND3 / 'finalize_report.csv', index=False)
    print(f'\nFinal round-3 expansion: {len(final)} proteins')
    print(f"  by move: {final['move'].value_counts().to_dict()}")
    print(f"  by domain: {final['domain'].value_counts().to_dict()}")
    print(f"  unique target families: {final['target_protein_family'].nunique()}")
    print(f"  unique species: {final['species'].nunique()}")
    print(f"  ribosomal proteins (should be ~0): {(final['broad_function'] == 'ribosomal').sum()}")
    print(f'Wrote {out}')
    print(f"Wrote {ROUND3 / 'finalize_report.csv'}")
fetch_and_annotate_round3_STAGES = {'queries': fetch_and_annotate_round3_stage_queries, 'fetch': fetch_and_annotate_round3_stage_fetch, 'filter': fetch_and_annotate_round3_stage_filter, 'structures': fetch_and_annotate_round3_stage_structures, 'features': fetch_and_annotate_round3_stage_features, 'classify': fetch_and_annotate_round3_stage_classify, 'finalize': fetch_and_annotate_round3_stage_finalize}
def fetch_and_annotate_round3_main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=list(fetch_and_annotate_round3_STAGES) + ['all'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    if args.stage == 'all':
        for name in ['queries', 'fetch', 'filter', 'structures', 'features', 'classify', 'finalize']:
            print(f'\n===== Stage: {name} =====')
            fn = fetch_and_annotate_round3_STAGES[name]
            kwargs = {}
            if name in ('queries', 'fetch'):
                kwargs['dry_run'] = args.dry_run
            if name in ('fetch', 'structures') and args.limit is not None:
                kwargs['limit'] = args.limit
            fn(**kwargs)
    else:
        fn = fetch_and_annotate_round3_STAGES[args.stage]
        kwargs = {}
        if args.stage in ('queries', 'fetch'):
            kwargs['dry_run'] = args.dry_run
        if args.stage in ('fetch', 'structures') and args.limit is not None:
            kwargs['limit'] = args.limit
        fn(**kwargs)
def fetch_and_annotate_round3__entry():
    sys.path.insert(0, str(fetch_and_annotate_round3_ROOT))
    ROUND3.mkdir(exist_ok=True)
    fetch_and_annotate_round3_PDB_CACHE.mkdir(exist_ok=True)
    fetch_and_annotate_round3_main()

# ---------- from build_round3_family_targets.py ----------
build_round3_family_targets_HERE = Path(__file__).parent
MAIN_PATH = build_round3_family_targets_HERE / 'Decoding_Bias_Dataset_updated.csv'
OUT = build_round3_family_targets_HERE / 'expansion_round3_family_targets.csv'
build_round3_family_targets_DOMAIN_TAXID = {'Archaea': 2157, 'Bacteria': 2, 'Eukaryota': 2759}
ALL_DOMAINS = list(build_round3_family_targets_DOMAIN_TAXID.keys())
MOVE_A_PER_MISSING_DOMAIN = 5
MOVE_A_MIN_CURRENT_MEMBERS = 5
MOVE_B_TARGET_TOTAL = 80
MOVE_B_SIZE_RANGE = (30, 120)
MOVE_C_UNIVERSAL_MIN_MEMBERS = 50
MOVE_C_MAX_PARALOGS_PER_SPECIES = 1
MOVE_C_MAX_PER_FAMILY = 40
build_round3_family_targets_BASE_FILTER = 'reviewed:true AND length:[50 TO 1000] AND NOT keyword:KW-0689'
def build_query(family_name, domain=None, species_id=None):
    parts = [f'({build_round3_family_targets_BASE_FILTER})', f'family:"{family_name}"']
    if domain:
        parts.append(f'taxonomy_id:{build_round3_family_targets_DOMAIN_TAXID[domain]}')
    if species_id:
        parts.append(f'organism_id:{int(species_id)}')
    return ' AND '.join(parts)
def build_round3_family_targets_main():
    df = pd.read_csv(MAIN_PATH)
    print(f"Loaded {len(df)} proteins, {df['protein_family'].nunique()} families from main.")
    fam = df.groupby('protein_family').agg(n_members=('Entry', 'size'), n_species=('species', 'nunique'), n_domains=('domain', 'nunique'), dominant_function=('broad_function', lambda x: x.mode().iloc[0]))
    fam_domains = df.groupby('protein_family')['domain'].agg(lambda x: set(x.dropna().unique()))
    fam['domains_present'] = fam_domains
    fam = fam.reset_index()
    fam = fam[fam['protein_family'] != 'Unclassified'].copy()
    rows = []
    move_a = fam[(fam['n_domains'] == 1) & (fam['n_members'] >= MOVE_A_MIN_CURRENT_MEMBERS)]
    print(f'\nMove A - single-domain promotion: {len(move_a)} families')
    for (_, r) in move_a.iterrows():
        missing = [d for d in ALL_DOMAINS if d not in r['domains_present']]
        for d in missing:
            rows.append({'move': 'A_multidomain', 'protein_family': r['protein_family'], 'current_n_members': int(r['n_members']), 'current_n_domains': int(r['n_domains']), 'current_n_species': int(r['n_species']), 'target_domain': d, 'target_n': MOVE_A_PER_MISSING_DOMAIN, 'query': build_query(r['protein_family'], domain=d), 'rationale': f"promote {r['n_members']}-member single-domain family to {d}"})
    move_b = fam[(fam['dominant_function'] != 'ribosomal') & (fam['n_members'] >= MOVE_B_SIZE_RANGE[0]) & (fam['n_members'] <= MOVE_B_SIZE_RANGE[1])]
    print(f'Move B - non-ribosomal big-family boost: {len(move_b)} families')
    for (_, r) in move_b.iterrows():
        deficit = max(0, MOVE_B_TARGET_TOTAL - int(r['n_members']))
        if deficit == 0:
            continue
        present = sorted(r['domains_present'])
        per_dom = max(1, deficit // len(present))
        for d in present:
            rows.append({'move': 'B_bignonribo', 'protein_family': r['protein_family'], 'current_n_members': int(r['n_members']), 'current_n_domains': int(r['n_domains']), 'current_n_species': int(r['n_species']), 'target_domain': d, 'target_n': per_dom, 'query': build_query(r['protein_family'], domain=d), 'rationale': f"boost {r['dominant_function']} family from {r['n_members']} → ~{MOVE_B_TARGET_TOTAL}"})
    move_c = fam[(fam['n_domains'] == 3) & (fam['n_members'] >= MOVE_C_UNIVERSAL_MIN_MEMBERS) & (fam['dominant_function'] != 'ribosomal')]
    print(f'Move C - universal non-ribo density: {len(move_c)} families')
    for (_, r) in move_c.iterrows():
        sub = df[df['protein_family'] == r['protein_family']]
        sp_counts = sub['species'].value_counts()
        single_member_species = sp_counts[sp_counts == 1].head(MOVE_C_MAX_PER_FAMILY)
        rows.append({'move': 'C_density', 'protein_family': r['protein_family'], 'current_n_members': int(r['n_members']), 'current_n_domains': int(r['n_domains']), 'current_n_species': int(r['n_species']), 'target_domain': 'any', 'target_n': min(len(single_member_species), MOVE_C_MAX_PER_FAMILY), 'query': build_query(r['protein_family']), 'rationale': f"add paralogs in {len(single_member_species)} single-member species (universal {r['dominant_function']} family)"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'\nWrote {OUT}: {len(out)} target rows')
    print(f"  Total target_n across all moves: {out['target_n'].sum()}")
    print(f'  Breakdown by move:')
    print(out.groupby('move').agg(rows=('target_n', 'size'), total_target=('target_n', 'sum')).to_string())
    print(f'\n  Breakdown by target_domain (Move A + B only):')
    ab = out[out['move'].isin(['A_multidomain', 'B_bignonribo'])]
    print(ab.groupby('target_domain')['target_n'].sum().to_string())
    print(f'\n  Sample queries:')
    for (_, r) in out.head(5).iterrows():
        print(f"    [{r['move']}] {r['protein_family'][:60]} → {r['target_domain']} (n={r['target_n']})")
        print(f"      query: {r['query']}")
def build_round3_family_targets__entry():
    build_round3_family_targets_main()

# ---------- from inject_round2_scores.py ----------
inject_round2_scores_HERE = Path(__file__).parent
NS = Path('/Users/lauradillon/New_scores')
COMBINED_PATH = inject_round2_scores_HERE / 'main_plus_round2.csv'
OUT_PATH = inject_round2_scores_HERE / 'main_plus_round2_scored.csv'
SCORE_MAP = {NS / 'proteinMPNN_results_all_20260515_184855.csv': ('sequence_score', 'proteinmpnn_score'), NS / '20260516_085537_esmif_results.csv': ('valid_pos_score', 'esmif_score'), NS / 'mif_likelihoods_final_20260516_092331.csv': ('MIF_Likelihood', 'mif_score'), NS / 'mif_likelihoods_final_20260516_143140.csv': ('MIF_Likelihood', 'mifst_score')}
def inject_round2_scores_main():
    combined = pd.read_csv(COMBINED_PATH, low_memory=False)
    round2_entries = set(combined.loc[combined['source'] == 'expansion_round2', 'Entry'])
    print(f'Loaded {len(combined)} rows; round-2 entries to fill: {len(round2_entries)}')
    for (path, (src_col, dst_col)) in SCORE_MAP.items():
        scores = pd.read_csv(path)
        if src_col not in scores.columns or 'Entry' not in scores.columns:
            print(f'  SKIP {path.name}: missing column {src_col} or Entry')
            continue
        score_map = scores.set_index('Entry')[src_col]
        used = score_map.index.intersection(round2_entries)
        unused = len(score_map) - len(used)
        mask = combined['Entry'].isin(used)
        combined.loc[mask, dst_col] = combined.loc[mask, 'Entry'].map(score_map).values
        n_filled = combined.loc[combined['source'] == 'expansion_round2', dst_col].notna().sum()
        print(f'  {dst_col:25s}  from {path.name}: used {len(used)}/{len(score_map)} (unused: {unused}); round-2 filled now: {n_filled}/{len(round2_entries)}')
    combined.to_csv(OUT_PATH, index=False)
    print(f'\nWrote {OUT_PATH}')
    print(f'  Score coverage on round-2 (1,362 rows):')
    r2 = combined[combined['source'] == 'expansion_round2']
    for col in ['proteinmpnn_score', 'esmif_score', 'mif_score', 'mifst_score', 'ESM2_15B_pppl_score', 'carp_640M_score']:
        print(f'    {col:25s} {r2[col].notna().sum():>5} / {len(r2)}')
def inject_round2_scores__entry():
    inject_round2_scores_main()

_STEPS = {
    'fetch-and-annotate-round2': fetch_and_annotate_round2__entry,
    'fetch-and-annotate-round3': fetch_and_annotate_round3__entry,
    'build-round3-family-targets': build_round3_family_targets__entry,
    'inject-round2-scores': inject_round2_scores__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

