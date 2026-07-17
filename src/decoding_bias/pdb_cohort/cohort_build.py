"""decoding_bias.pdb_cohort.cohort_build -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - build_independent_pdb_cohort
  - build_mif_safe_cohort
  - scan_uniprot_pdb
  - implement_uniprot_pdb_retrieval
  - prepare_pdb_chain_sequences
"""

from __future__ import annotations

import argparse
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio
import json
import numpy as np
import pandas as pd
import re
import requests
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from io import StringIO
from pathlib import Path
from dataset_update.protein_classification import classify
from dataset_update.collapse_species_subspecies import collapse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from tqdm import tqdm
from decoding_bias.features.structural_features import calculate_avg_cb_distance, calculate_compactness, calculate_contact_order, calculate_surface_exposure, download_alphafold_structure, extract_plddt_scores, extract_secondary_structure, parse_structure, scan_existing_structures

# ---------- from build_independent_pdb_cohort.py ----------
build_independent_pdb_cohort_HERE = Path(__file__).resolve().parent
build_independent_pdb_cohort_ROOT = build_independent_pdb_cohort_HERE.parent
build_independent_pdb_cohort_OUT = build_independent_pdb_cohort_HERE / 'outputs' / 'independent_cohort'
SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
GRAPHQL_URL = 'https://data.rcsb.org/graphql'
build_independent_pdb_cohort_UNIPROT_SEARCH = 'https://rest.uniprot.org/uniprotkb/search'
MAIN_META = build_independent_pdb_cohort_ROOT / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
DOMAINS = ['Bacteria', 'Eukaryota', 'Archaea']
build_independent_pdb_cohort_UNIPROT_FIELDS = ','.join(['accession', 'protein_name', 'protein_families', 'xref_pfam', 'ec', 'go_f', 'ft_transmem', 'ft_carbohyd', 'xref_pdb', 'xref_disprot', 'xref_ideal', 'lineage', 'organism_name', 'length'])
CLASSIFY_RENAME = {'Protein names': 'Protein names', 'Protein families': 'Protein families', 'Pfam': 'Pfam', 'EC number': 'EC number', 'Gene Ontology (molecular function)': 'Gene Ontology (molecular function)', 'Transmembrane': 'Transmembrane', 'Glycosylation': 'Glycosylation', 'PDB': 'PDB', 'DisProt': 'DisProt', 'IDEAL': 'IDEAL'}
STD_AA = set('ACDEFGHIKLMNPQRSTVWY')
def _get_json(url: str, retries: int=4, timeout: int=90) -> dict:
    last = None
    for attempt in range(1, retries + 1):
        try:
            return json.load(urllib.request.urlopen(url, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return {}
            last = e
            time.sleep(min(2 * attempt, 12))
        except Exception as e:
            last = e
            time.sleep(min(2 * attempt, 12))
    raise RuntimeError(f'GET failed after {retries} tries: {last}')
def _post_json(url: str, payload: dict, retries: int=4, timeout: int=120) -> dict:
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception as e:
            last = e
            time.sleep(min(2 * attempt, 12))
    raise RuntimeError(f'POST failed after {retries} tries: {last}')
def domain_from_lineage(lineage: str | None) -> str | None:
    """First superkingdom token found in a UniProt 'Taxonomic lineage' string."""
    if not isinstance(lineage, str):
        return None
    for d in DOMAINS:
        if re.search(f'\\b{d}\\b', lineage):
            return d
    return None
def build_query(args, start: int, rows: int) -> dict:
    return {'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [{'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'exptl.method', 'operator': 'exact_match', 'value': 'X-RAY DIFFRACTION'}}, {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'rcsb_entry_info.resolution_combined', 'operator': 'less_or_equal', 'value': args.resolution}}, {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'entity_poly.rcsb_entity_polymer_type', 'operator': 'exact_match', 'value': 'Protein'}}, {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'entity_poly.rcsb_sample_sequence_length', 'operator': 'range', 'value': {'from': args.min_len, 'to': args.max_len, 'include_lower': True, 'include_upper': True}}}, {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'rcsb_entity_source_organism.taxonomy_lineage.name', 'operator': 'in', 'value': DOMAINS}}, {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'rcsb_assembly_info.polymer_entity_instance_count', 'operator': 'equals', 'value': 1}}]}, 'return_type': 'polymer_entity', 'request_options': {'group_by': {'aggregation_method': 'sequence_identity', 'similarity_cutoff': args.identity}, 'group_by_return_type': 'representatives', 'paginate': {'start': start, 'rows': rows}}}
def stage_search(args) -> None:
    build_independent_pdb_cohort_OUT.mkdir(parents=True, exist_ok=True)
    rows_per = 1000
    q = build_query(args, 0, rows_per)
    url = SEARCH_URL + '?json=' + urllib.parse.quote(json.dumps(q))
    out = _get_json(url)
    total = out.get('total_count')
    n_groups = out.get('group_by_count')
    print(f'matching entities: {total:,}  |  30%-id representatives: {n_groups:,}')
    ids = [r['identifier'] for r in out.get('result_set', [])]
    start = rows_per
    while start < n_groups:
        q = build_query(args, start, rows_per)
        url = SEARCH_URL + '?json=' + urllib.parse.quote(json.dumps(q))
        page = _get_json(url)
        batch = [r['identifier'] for r in page.get('result_set', [])]
        if not batch:
            break
        ids.extend(batch)
        start += rows_per
        print(f'  collected {len(ids):,}/{n_groups:,} representatives', end='\r')
        time.sleep(0.1)
    print()
    df = pd.DataFrame({'entity_id': ids})
    df.to_csv(build_independent_pdb_cohort_OUT / 'candidates_representatives.csv', index=False)
    print(f"wrote {build_independent_pdb_cohort_OUT / 'candidates_representatives.csv'}  ({len(df):,} rows)")
GRAPHQL_Q = '\n{ polymer_entities(entity_ids: %s) {\n    rcsb_id\n    entity_poly { pdbx_seq_one_letter_code_can rcsb_sample_sequence_length }\n    rcsb_polymer_entity { pdbx_description }\n    rcsb_polymer_entity_container_identifiers { entry_id uniprot_ids auth_asym_ids }\n    rcsb_entity_source_organism { ncbi_scientific_name ncbi_taxonomy_id }\n    entry { rcsb_entry_info { resolution_combined } }\n} }'
def fetch_graphql(entity_ids: list[str], batch: int=50) -> pd.DataFrame:
    rows = []
    for i in range(0, len(entity_ids), batch):
        chunk = entity_ids[i:i + batch]
        payload = {'query': GRAPHQL_Q % json.dumps(chunk)}
        out = _post_json(GRAPHQL_URL, payload)
        for e in out.get('data', {}).get('polymer_entities') or []:
            if e is None:
                continue
            ci = e.get('rcsb_polymer_entity_container_identifiers') or {}
            poly = e.get('entity_poly') or {}
            org = (e.get('rcsb_entity_source_organism') or [{}])[0] or {}
            entry = (e.get('entry') or {}).get('rcsb_entry_info') or {}
            res = entry.get('resolution_combined')
            uni = ci.get('uniprot_ids') or []
            chains = ci.get('auth_asym_ids') or []
            rows.append({'entity_id': e['rcsb_id'], 'pdb_id': ci.get('entry_id'), 'chain': chains[0] if chains else None, 'uniprot': uni[0] if uni else None, 'sequence': poly.get('pdbx_seq_one_letter_code_can'), 'Length': poly.get('rcsb_sample_sequence_length'), 'pdb_description': (e.get('rcsb_polymer_entity') or {}).get('pdbx_description'), 'organism_rcsb': org.get('ncbi_scientific_name'), 'ncbi_taxid': org.get('ncbi_taxonomy_id'), 'resolution_A': res[0] if isinstance(res, list) and res else res})
        print(f'  graphql {min(i + batch, len(entity_ids)):,}/{len(entity_ids):,}', end='\r')
        time.sleep(0.05)
    print()
    return pd.DataFrame(rows)
def fetch_uniprot(accessions: list[str], batch: int=100) -> pd.DataFrame:
    rows = []
    for i in range(0, len(accessions), batch):
        chunk = accessions[i:i + batch]
        query = ' OR '.join((f'accession:{a}' for a in chunk))
        params = urllib.parse.urlencode({'query': query, 'format': 'tsv', 'fields': build_independent_pdb_cohort_UNIPROT_FIELDS, 'size': len(chunk)})
        try:
            txt = urllib.request.urlopen(f'{build_independent_pdb_cohort_UNIPROT_SEARCH}?{params}', timeout=90).read().decode()
            page = pd.read_csv(StringIO(txt), sep='\t')
            if len(page):
                rows.append(page)
        except Exception as e:
            print(f'  uniprot batch {i} failed: {e}')
        print(f'  uniprot {min(i + batch, len(accessions)):,}/{len(accessions):,}', end='\r')
        time.sleep(0.1)
    print()
    uni = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(uni):
        uni = uni.drop_duplicates('Entry', keep='first')
    return uni
def stage_annotate(args) -> None:
    cand = pd.read_csv(build_independent_pdb_cohort_OUT / 'candidates_representatives.csv')
    ids = cand['entity_id'].tolist()
    gcache = build_independent_pdb_cohort_OUT / '_graphql_raw.csv'
    if gcache.exists() and (not args.refetch):
        print(f'reusing cached GraphQL ({gcache})')
        g = pd.read_csv(gcache)
    else:
        print(f'annotating {len(ids):,} representatives via GraphQL...')
        g = fetch_graphql(ids)
        g.to_csv(gcache, index=False)
    g = g[g['sequence'].notna()].copy()
    g['sequence'] = g['sequence'].str.upper().str.replace('U', 'C', regex=False)
    g['nonstd'] = g['sequence'].apply(lambda s: bool(set(s) - STD_AA))
    g = g[~g['nonstd']].copy()
    g = g[(g['Length'] >= args.min_len) & (g['Length'] <= args.max_len)]
    print(f'  {len(g):,} pass standard-AA + length')
    accs = sorted(g['uniprot'].dropna().unique().tolist())
    ucache = build_independent_pdb_cohort_OUT / '_uniprot_raw.tsv'
    if ucache.exists() and (not args.refetch):
        print(f'reusing cached UniProt ({ucache})')
        uni = pd.read_csv(ucache, sep='\t', low_memory=False)
    else:
        print(f'fetching UniProt fields for {len(accs):,} accessions...')
        uni = fetch_uniprot(accs)
        uni.to_csv(ucache, sep='\t', index=False)
    cls_in = uni.rename(columns=CLASSIFY_RENAME)
    for col in CLASSIFY_RENAME.values():
        if col not in cls_in.columns:
            cls_in[col] = pd.NA
    classified = classify(cls_in)[['Entry', 'protein_name_clean', 'protein_family', 'broad_function', 'is_enzyme', 'is_transmembrane', 'is_glycosylated', 'has_disordered']].rename(columns={'Entry': 'uniprot'})
    lin = uni.rename(columns={'Taxonomic lineage': 'lineage'}) if 'Taxonomic lineage' in uni else uni
    lin = lin[['Entry', 'lineage', 'Organism']].rename(columns={'Entry': 'uniprot', 'Organism': 'organism_uniprot'})
    lin['domain'] = lin['lineage'].apply(domain_from_lineage)
    df = g.merge(classified, on='uniprot', how='left').merge(lin, on='uniprot', how='left')
    df['species'] = df['organism_uniprot'].fillna(df['organism_rcsb'])
    df['species'] = df['species'].apply(lambda x: re.sub('\\s*\\([^)]*\\)', '', x).strip() if isinstance(x, str) else x)
    df['species_collapsed'] = df['species'].apply(collapse)
    df['broad_function'] = df['broad_function'].fillna('other')
    df['protein_family'] = df['protein_family'].fillna('Unclassified')
    n_nodomain = df['domain'].isna().sum()
    df = df[df['domain'].notna()].copy()
    print(f'  dropped {n_nodomain:,} without resolvable domain; {len(df):,} annotated')
    df.to_csv(build_independent_pdb_cohort_OUT / 'annotated.csv', index=False)
    print(f"wrote {build_independent_pdb_cohort_OUT / 'annotated.csv'}  ({len(df):,} rows)")
    print('\ndomain distribution (annotated pool):')
    print(df['domain'].value_counts())
    print('\nbroad_function (top 12):')
    print(df['broad_function'].value_counts().head(12))
def breadth_filter(df: pd.DataFrame, min_species_per_family: int=5, min_proteins_per_species: int=2) -> pd.DataFrame:
    """Iteratively enforce: each species has >=2 proteins AND each protein_family
    is represented by >=5 distinct species, within the cohort's own taxonomy."""
    cur = df.copy()
    while True:
        sp_counts = cur.groupby('species_collapsed')['entity_id'].transform('count')
        cur = cur[sp_counts >= min_proteins_per_species]
        fam_species = cur.groupby('protein_family')['species_collapsed'].transform('nunique')
        cur2 = cur[fam_species >= min_species_per_family]
        if len(cur2) == len(cur):
            return cur2
        cur = cur2
def stage_match(args) -> None:
    df = pd.read_csv(build_independent_pdb_cohort_OUT / 'annotated.csv')
    print(f'annotated pool: {len(df):,}')
    df = breadth_filter(df)
    print(f'after breadth filter (family>=5 species, species>=2): {len(df):,}')
    print(df['domain'].value_counts())
    meta = pd.read_csv(MAIN_META, low_memory=False)
    target = meta['domain'].value_counts(normalize=True)
    print('\ntarget (full v12) domain marginal:')
    print(target.round(3))
    avail = df['domain'].value_counts()
    max_n = min((int(avail.get(d, 0) / target.get(d, 1e-09)) for d in target.index if target.get(d, 0) > 0))
    n = args.target_n if args.target_n > 0 else max_n
    n = min(n, max_n)
    print(f'\nmax cohort at exact domain marginal: {max_n:,}  -> building N={n:,}')
    parts = []
    rng_seed = args.seed
    for (d, frac) in target.items():
        k = int(round(frac * n))
        sub = df[df['domain'] == d]
        k = min(k, len(sub))
        parts.append(sub.sample(n=k, random_state=rng_seed))
    cohort = pd.concat(parts, ignore_index=True)
    cohort.to_csv(build_independent_pdb_cohort_OUT / 'cohort_manifest.csv', index=False)
    print(f"\nwrote {build_independent_pdb_cohort_OUT / 'cohort_manifest.csv'}  ({len(cohort):,} proteins)")
    print('cohort domain mix:')
    print(cohort['domain'].value_counts(normalize=True).round(3))
    print('cohort broad_function (top 12; ribosomal is best-effort):')
    print(cohort['broad_function'].value_counts().head(12))
    cohort[['entity_id', 'pdb_id', 'chain', 'uniprot', 'sequence', 'Length', 'domain', 'species_collapsed', 'protein_family', 'broad_function', 'resolution_A']].to_csv(build_independent_pdb_cohort_OUT / 'cohort_scoring_inputs.csv', index=False)
    cohort[['pdb_id', 'chain', 'entity_id']].drop_duplicates().to_csv(build_independent_pdb_cohort_OUT / 'cohort_structures_to_download.csv', index=False)
    print(f'wrote cohort_scoring_inputs.csv + cohort_structures_to_download.csv')
def build_independent_pdb_cohort_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['search', 'annotate', 'match', 'all'])
    ap.add_argument('--resolution', type=float, default=2.5)
    ap.add_argument('--min-len', type=int, default=50)
    ap.add_argument('--max-len', type=int, default=1000)
    ap.add_argument('--identity', type=int, default=30, choices=[30, 40, 50, 70, 90, 95, 100])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target-n', type=int, default=0, help='0 = max possible at exact domain marginal')
    ap.add_argument('--refetch', action='store_true', help='ignore cached GraphQL/UniProt files and re-download')
    args = ap.parse_args()
    if args.stage in ('search', 'all'):
        stage_search(args)
    if args.stage in ('annotate', 'all'):
        stage_annotate(args)
    if args.stage in ('match', 'all'):
        stage_match(args)
def build_independent_pdb_cohort__entry():
    sys.path.insert(0, str(build_independent_pdb_cohort_ROOT))
    build_independent_pdb_cohort_main()

# ---------- from build_mif_safe_cohort.py ----------
build_mif_safe_cohort_HERE = Path(__file__).resolve().parent
COH = build_mif_safe_cohort_HERE / 'outputs' / 'independent_cohort'
IN_CSV = COH / 'cohort_pdb_scoring_inputs.csv'
BACKBONE = {'N', 'CA', 'C'}
def check(pdb_path: str, seq_len: int) -> str | None:
    """Return None if MIF-safe, else a short reason string."""
    try:
        arr = pdbio.PDBFile.read(pdb_path).get_structure(model=1)
    except Exception as ex:
        return f'parse_error:{str(ex)[:30]}'
    aa = arr[struc.filter_amino_acids(arr)]
    if aa.array_length() == 0:
        return 'no_amino_acids'
    if np.any(aa.ins_code != ''):
        return 'insertion_codes'
    res_ids = aa.res_id[struc.get_residue_starts(aa)]
    n_res = len(res_ids)
    if n_res != seq_len:
        return f'residue_count_{n_res}!=seq_{seq_len}'
    span = int(res_ids.max() - res_ids.min() + 1)
    if span != n_res:
        return f'numbering_gap_span_{span}!=n_{n_res}'
    starts = struc.get_residue_starts(aa, add_exclusive_stop=True)
    for i in range(len(starts) - 1):
        names = set(aa.atom_name[starts[i]:starts[i + 1]])
        if not BACKBONE.issubset(names):
            return 'incomplete_backbone'
    return None
def build_mif_safe_cohort_main():
    df = pd.read_csv(IN_CSV)
    reasons = []
    for r in df.itertuples(index=False):
        reasons.append(check(str(r.chain_pdb_path), len(str(r.sequence))))
    df = df.copy()
    df['mif_fail_reason'] = reasons
    df.to_csv(COH / 'cohort_pdb_scoring_inputs_mif_qc.csv', index=False)
    safe = df[df['mif_fail_reason'].isna()].drop(columns=['mif_fail_reason'])
    safe.to_csv(COH / 'cohort_pdb_scoring_inputs_mif_safe.csv', index=False)
    print(f'total: {len(df)}  |  MIF-safe: {len(safe)} ({len(safe) / len(df) * 100:.0f}%)')
    print('exclusion reasons (categorised):')
    cat = df['mif_fail_reason'].dropna().str.replace('_.*', '', regex=True)
    print(cat.value_counts().to_string())
    print('\nMIF-safe by domain:')
    print(safe['domain'].value_counts().to_string())
    print(f'\nwrote cohort_pdb_scoring_inputs_mif_safe.csv + _mif_qc.csv')
def build_mif_safe_cohort__entry():
    warnings.filterwarnings('ignore')
    build_mif_safe_cohort_main()

# ---------- from scan_uniprot_pdb.py ----------
REPO = Path(__file__).resolve().parent.parent
ANA = REPO / 'dataset_update' / 'main_plus_r2_r3_analysis_v12_corrected.csv'
META = REPO / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
scan_uniprot_pdb_OUT = Path(__file__).resolve().parent / 'outputs'
CACHE = scan_uniprot_pdb_OUT / '_uniprot_pdb_xref_cache.json'
PARTIAL = scan_uniprot_pdb_OUT / 'pdb_availability.partial.csv'
METHOD_RANK = {'X-ray': 3, 'EM': 2, 'NMR': 1}
def parse_chains(s):
    """'O/P/Q/R=1-335, A=10-50' -> [(chain, start, end), ...] (first chain per group)."""
    out = []
    for grp in s.split(','):
        grp = grp.strip()
        if '=' not in grp:
            continue
        (chains, rng) = grp.split('=', 1)
        m = re.match('(-?\\d+)-(-?\\d+)', rng.strip())
        if not m:
            continue
        first_chain = chains.split('/')[0].strip()
        out.append((first_chain, int(m.group(1)), int(m.group(2))))
    return out
def best_pdb(acc, uni_len, cache):
    if acc in cache:
        recs = cache[acc]
    else:
        try:
            d = json.load(urllib.request.urlopen(f'https://rest.uniprot.org/uniprotkb/{acc}.json', timeout=25))
            recs = []
            for x in d.get('uniProtKBCrossReferences', []):
                if x['database'] != 'PDB':
                    continue
                p = {q['key']: q['value'] for q in x.get('properties', [])}
                recs.append({'id': x['id'], 'method': p.get('Method', ''), 'res': p.get('Resolution', ''), 'chains': p.get('Chains', '')})
            cache[acc] = recs
        except Exception as ex:
            cache[acc] = []
            recs = []
        time.sleep(0.02)
    best = None
    for r in recs:
        for (ch, s, e) in parse_chains(r['chains']):
            cov = (e - s + 1) / uni_len if uni_len else 0
            mres = re.match('([\\d.]+)', str(r['res']))
            resol = float(mres.group(1)) if mres else 99.0
            key = (round(cov, 3), METHOD_RANK.get(r['method'], 0), -resol)
            cand = dict(best_pdb_id=r['id'], best_chain=ch, best_method=r['method'], best_resolution_A=resol if resol < 99 else None, cov_start=s, cov_end=e, coverage_frac=round(cov, 3), _key=key)
            if best is None or key > best['_key']:
                best = cand
    return (len(recs), best)
def scan_uniprot_pdb_main():
    ana = pd.read_csv(ANA, low_memory=False)[['Entry', 'sequence_length', 'domain']]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    if PARTIAL.exists():
        rows = pd.read_csv(PARTIAL).to_dict('records')
        done = set((r['Entry'] for r in rows))
        ana = ana[~ana.Entry.isin(done)]
        print(f'resuming, {len(done)} done')
    recs = ana.to_dict('records')
    from tqdm import tqdm

    def work(r):
        (n, b) = best_pdb(r['Entry'], r['sequence_length'], cache)
        rec = dict(Entry=r['Entry'], domain=r['domain'], uniprot_len=r['sequence_length'], n_pdb=n, has_pdb=b is not None)
        if b:
            b.pop('_key')
            rec.update(b)
        return rec
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(work, r) for r in recs]
        for (i, f) in enumerate(tqdm(as_completed(futs), total=len(futs), desc='UniProt scan')):
            rows.append(f.result())
            if i % 200 == 0:
                pd.DataFrame(rows).to_csv(PARTIAL, index=False)
                CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))
    df = pd.DataFrame(rows)
    df.to_csv(scan_uniprot_pdb_OUT / 'pdb_availability.csv', index=False)
    if PARTIAL.exists():
        PARTIAL.unlink()
    hp = df[df.has_pdb]
    print(f'\nFresh scan: {df.has_pdb.sum()}/{len(df)} proteins have >=1 PDB (UniProt xref)')
    print('By domain (has PDB):')
    print(hp.domain.value_counts().to_string())
    print('Method of best structure:')
    print(hp.best_method.value_counts().to_string())
    print(f'Median coverage of best structure: {hp.coverage_frac.median():.2f}')
    print(f'With >=90% coverage: {(hp.coverage_frac >= 0.9).sum()}')
    old = pd.read_csv(META, low_memory=False)[['Entry', 'has_pdb_struct']]
    cmp = df.merge(old, on='Entry', how='left')
    print(f'\nvs old metadata has_pdb_struct: old={int(cmp.has_pdb_struct.sum())}, fresh={int(cmp.has_pdb.sum())}')
    print(f'  fresh-only (old missed): {int((cmp.has_pdb & ~cmp.has_pdb_struct.fillna(False)).sum())}')
    print(f'  old-only (false positives): {int((~cmp.has_pdb & cmp.has_pdb_struct.fillna(False)).sum())}')
    print(f'\nWrote pdb_availability.csv')
def scan_uniprot_pdb__entry():
    warnings.filterwarnings('ignore')
    scan_uniprot_pdb_main()

# ---------- from implement_uniprot_pdb_retrieval.py ----------
implement_uniprot_pdb_retrieval_HERE = Path(__file__).resolve().parent
implement_uniprot_pdb_retrieval_ROOT = implement_uniprot_pdb_retrieval_HERE.parent
DEFAULT_INPUT = implement_uniprot_pdb_retrieval_HERE / 'main_plus_r2_r3_scored_filterC_v7.csv'
DEFAULT_OUTPUT = implement_uniprot_pdb_retrieval_HERE / 'main_plus_r2_r3_scored_filterC_v8.csv'
DEFAULT_OUTDIR = implement_uniprot_pdb_retrieval_HERE / 'retrieval_recipe_outputs'
implement_uniprot_pdb_retrieval_UNIPROT_SEARCH = 'https://rest.uniprot.org/uniprotkb/search'
implement_uniprot_pdb_retrieval_UNIPROT_FIELDS = ','.join(['accession', 'id', 'protein_name', 'gene_names', 'organism_name', 'organism_id', 'lineage', 'length', 'sequence', 'ec', 'keyword', 'cc_function', 'cc_subcellular_location', 'cc_subunit', 'xref_pfam', 'protein_families', 'xref_pdb', 'ft_transmem', 'ft_carbohyd', 'xref_disprot', 'xref_ideal', 'go_f', 'xref_alphafolddb'])
UNIPROT_RENAME = {'Entry Name': 'EntryName', 'Protein names': 'protein_name', 'Gene Names': 'gene_names', 'Organism': 'Organism', 'Organism (ID)': 'organism_id', 'Taxonomic lineage': 'Taxonomic lineage', 'Length': 'Length', 'Sequence': 'sequence', 'EC number': 'ec', 'Keywords': 'keywords', 'Function [CC]': 'cc_function', 'Subcellular location [CC]': 'cc_subcellular_location', 'Subunit structure': 'Subunit structure', 'Pfam': 'Pfam', 'Protein families': 'protein_families_raw', 'PDB': 'pdb_ids_raw', 'Transmembrane': 'Transmembrane', 'Glycosylation': 'Glycosylation', 'DisProt': 'DisProt', 'IDEAL': 'IDEAL', 'Gene Ontology (molecular function)': 'Gene Ontology (molecular function)', 'AlphaFoldDB': 'AlphaFoldDB'}
DIRECT_UNIPROT_FILL_COLUMNS = ['EntryName', 'protein_name', 'gene_names', 'Organism', 'organism_id', 'Taxonomic lineage', 'Length', 'sequence', 'ec', 'keywords', 'cc_function', 'cc_subcellular_location', 'Subunit structure', 'Pfam', 'protein_families_raw', 'pdb_ids_raw']
CLASSIFIER_FILL_COLUMNS = ['protein_name_clean', 'protein_family', 'broad_function', 'is_enzyme', 'is_transmembrane', 'is_glycosylated', 'has_disordered', 'has_pdb']
STRUCTURE_DERIVED_COLUMNS = ['avg_plddt', 'min_plddt', 'max_plddt', 'plddt_very_high_pct', 'plddt_high_pct', 'plddt_medium_pct', 'plddt_low_pct', 'surface_exposure', 'helix_percent', 'sheet_percent', 'loop_percent', 'helix_sheet_contrast', 'ordered_percent', 'avg_cb_distance', 'compactness', 'structural_compactness', 'centralization', 'rco']
NULL_STRINGS = {'', 'nan', 'NaN', 'None', 'NA', 'N/A', '<NA>'}
PDB_ID_RE = re.compile('\\b[0-9][A-Za-z0-9]{3}\\b')
RANK_RE = re.compile('\\s*([^,]+?)\\s+\\(([^)]+)\\)\\s*')
DOMAIN_NAMES = ('Viruses', 'Archaea', 'Bacteria', 'Eukaryota')
def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype('string').str.strip().isin(NULL_STRINGS)
def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in NULL_STRINGS:
        return True
    return False
def fill_missing_from_series(df: pd.DataFrame, values: pd.Series, column: str, counts: dict[str, int]) -> None:
    """Fill only missing target values from a series indexed by Entry."""
    if column not in df.columns:
        incoming_nonmissing = values.dropna()
        sample = incoming_nonmissing.iloc[0] if not incoming_nonmissing.empty else None
        dtype = 'object' if isinstance(sample, (str, bool, np.bool_)) else 'float64'
        df[column] = pd.Series([np.nan] * len(df), index=df.index, dtype=dtype)
    elif not pd.api.types.is_object_dtype(df[column].dtype):
        incoming_nonmissing = values.dropna()
        if not incoming_nonmissing.empty:
            sample = incoming_nonmissing.iloc[0]
            if isinstance(sample, (str, bool, np.bool_)):
                df[column] = df[column].astype('object')
    incoming = df['Entry'].map(values)
    fill = missing_mask(df[column]) & incoming.notna() & ~incoming.astype('string').str.strip().isin(NULL_STRINGS)
    df.loc[fill, column] = incoming.loc[fill]
    counts[column] = counts.get(column, 0) + int(fill.sum())
def batch_fetch_uniprot(entries: list[str], batch_size: int, sleep_s: float, retries: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch UniProt TSV rows for accessions in batches."""
    rows = []
    failures = []
    session = requests.Session()
    for start in tqdm(range(0, len(entries), batch_size), desc='UniProt batches'):
        batch = entries[start:start + batch_size]
        query = ' OR '.join((f'accession:{entry}' for entry in batch))
        params = {'query': query, 'format': 'tsv', 'fields': implement_uniprot_pdb_retrieval_UNIPROT_FIELDS, 'size': len(batch)}
        response = None
        error = ''
        for attempt in range(1, retries + 1):
            try:
                response = session.get(implement_uniprot_pdb_retrieval_UNIPROT_SEARCH, params=params, timeout=60)
                response.raise_for_status()
                break
            except Exception as exc:
                error = str(exc)
                response = None
                time.sleep(min(2 * attempt, 10))
        if response is None:
            for entry in batch:
                failures.append({'Entry': entry, 'reason': error or 'request_failed'})
            continue
        page = pd.read_csv(StringIO(response.text), sep='\t')
        if len(page):
            rows.append(page)
            got = set(page['Entry'].astype(str))
        else:
            got = set()
        for entry in batch:
            if entry not in got:
                failures.append({'Entry': entry, 'reason': 'not_returned_by_uniprot'})
        time.sleep(sleep_s)
    uni = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(uni):
        uni = uni.drop_duplicates('Entry', keep='first')
    return (uni, pd.DataFrame(failures))
def prepare_uniprot_for_merge(uni_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize UniProt column names and compute classifier-derived fields."""
    uni = uni_raw.rename(columns={k: v for (k, v) in UNIPROT_RENAME.items() if k in uni_raw.columns}).copy()
    classifier_input = uni_raw.rename(columns={'Protein names': 'Protein names', 'Protein families': 'Protein families', 'EC number': 'EC number', 'Transmembrane': 'Transmembrane', 'Glycosylation': 'Glycosylation', 'PDB': 'PDB', 'DisProt': 'DisProt', 'IDEAL': 'IDEAL', 'Gene Ontology (molecular function)': 'Gene Ontology (molecular function)'})
    classified = classify(classifier_input)
    classifier_keep = ['Entry'] + [c for c in CLASSIFIER_FILL_COLUMNS if c in classified.columns]
    classifier_df = classified[classifier_keep].copy()
    return (uni, classifier_df)
def parse_lineage_ranks(lineage: Any) -> dict[str, str | None]:
    out: dict[str, str | None] = {'domain': None, 'phylum_division': None, 'class': None, 'genus': None}
    if not isinstance(lineage, str) or not lineage.strip():
        return out
    for part in lineage.split(','):
        match = RANK_RE.match(part)
        if not match:
            continue
        (name, rank) = (match.group(1).strip(), match.group(2).strip().lower())
        if rank == 'domain' and name in DOMAIN_NAMES:
            out['domain'] = name
        elif rank in {'phylum', 'division'} and out['phylum_division'] is None:
            out['phylum_division'] = name
        elif rank == 'class' and out['class'] is None:
            out['class'] = name
        elif rank == 'genus' and out['genus'] is None:
            out['genus'] = name
    return out
def normalize_species(name: Any) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    cleaned = re.sub('\\s*\\(strain[^)]*\\)', '', name)
    cleaned = re.sub('\\s*\\([^)]+\\)', '', cleaned).strip()
    return cleaned or None
def first_pdb_id(raw_pdb: Any) -> str | None:
    if not isinstance(raw_pdb, str):
        return None
    match = PDB_ID_RE.search(raw_pdb)
    return match.group(0).upper() if match else None
def construct_description(row: pd.Series) -> str | None:
    entry = row.get('Entry')
    entry_name = row.get('EntryName')
    protein_name = row.get('protein_name')
    organism = row.get('Organism')
    organism_id = row.get('organism_id')
    gene_names = row.get('gene_names')
    if is_missing(entry) or is_missing(protein_name):
        return None
    prefix = f'sp|{entry}|{entry_name}' if not is_missing(entry_name) else str(entry)
    parts = [prefix, str(protein_name)]
    if not is_missing(organism):
        parts.append(f'OS={organism}')
    if not is_missing(organism_id):
        try:
            ox = str(int(float(organism_id)))
        except Exception:
            ox = str(organism_id)
        parts.append(f'OX={ox}')
    if not is_missing(gene_names):
        first_gene = str(gene_names).split()[0]
        parts.append(f'GN={first_gene}')
    return ' '.join(parts)
def apply_uniprot_fills(df: pd.DataFrame, uni: pd.DataFrame, classifier_df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    uni_by_entry = uni.set_index('Entry', drop=False)
    for col in DIRECT_UNIPROT_FILL_COLUMNS:
        if col in uni_by_entry.columns:
            fill_missing_from_series(df, uni_by_entry[col], col, counts)
    if 'Taxonomic lineage' in df.columns:
        fill_missing_from_series(df, df.set_index('Entry')['Taxonomic lineage'], 'lineage', counts)
    if 'lineage' in df.columns:
        fill_missing_from_series(df, df.set_index('Entry')['lineage'], 'Taxonomic lineage', counts)
    lineage_source = df['Taxonomic lineage'].where(~missing_mask(df['Taxonomic lineage']), df.get('lineage'))
    ranks = lineage_source.apply(parse_lineage_ranks).apply(pd.Series)
    for col in ['domain', 'phylum_division', 'class', 'genus']:
        if col in ranks.columns:
            fill_missing_from_series(df, pd.Series(ranks[col].values, index=df['Entry']), col, counts)
    if 'Organism' in df.columns:
        species = df['Organism'].apply(normalize_species)
        fill_missing_from_series(df, pd.Series(species.values, index=df['Entry']), 'species', counts)
    desc = df.apply(construct_description, axis=1)
    fill_missing_from_series(df, pd.Series(desc.values, index=df['Entry']), 'Description', counts)
    classifier_by_entry = classifier_df.set_index('Entry', drop=False)
    for col in CLASSIFIER_FILL_COLUMNS:
        if col in classifier_by_entry.columns:
            fill_missing_from_series(df, classifier_by_entry[col], col, counts)
    if 'pdb_ids_raw' in df.columns:
        first_ids = df['pdb_ids_raw'].apply(first_pdb_id)
        fill_missing_from_series(df, pd.Series(first_ids.values, index=df['Entry']), 'pdb_id', counts)
        has_pdb = df['pdb_ids_raw'].notna() & ~missing_mask(df['pdb_ids_raw'])
        if 'has_pdb' not in df.columns:
            df['has_pdb'] = np.nan
        fill = missing_mask(df['has_pdb']) & has_pdb
        df.loc[fill, 'has_pdb'] = True
        counts['has_pdb'] = counts.get('has_pdb', 0) + int(fill.sum())
        if 'has_pdb_struct' not in df.columns:
            df['has_pdb_struct'] = False
        before = df['has_pdb_struct'].fillna(False).astype(bool)
        after = before | has_pdb
        changed = after & ~before
        df.loc[changed, 'has_pdb_struct'] = True
        counts['has_pdb_struct'] = counts.get('has_pdb_struct', 0) + int(changed.sum())
    alias_pairs = [('target_domain', 'domain'), ('target_broad_function', 'broad_function'), ('target_protein_family', 'protein_family')]
    for (target, source) in alias_pairs:
        if source in df.columns:
            fill_missing_from_series(df, df.set_index('Entry')[source], target, counts)
    return counts
def load_structure_map(cache_dirs: list[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cache_dir in cache_dirs:
        for (entry, path) in scan_existing_structures(str(cache_dir)).items():
            mapping.setdefault(str(entry), path)
    return mapping
def structure_columns_missing(row: pd.Series) -> list[str]:
    return [c for c in STRUCTURE_DERIVED_COLUMNS if c in row.index and is_missing(row[c])]
def fill_cheap_structure_derivatives(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    derived = {'loop_percent': 1 - df['helix_percent'] - df['sheet_percent'] if {'helix_percent', 'sheet_percent'}.issubset(df.columns) else None, 'helix_sheet_contrast': df['helix_percent'] - df['sheet_percent'] if {'helix_percent', 'sheet_percent'}.issubset(df.columns) else None, 'ordered_percent': df['helix_percent'] + df['sheet_percent'] if {'helix_percent', 'sheet_percent'}.issubset(df.columns) else None, 'structural_compactness': 1 / df['compactness'] if 'compactness' in df.columns else None, 'centralization': 1 / df['avg_cb_distance'] if 'avg_cb_distance' in df.columns else None}
    for (col, values) in derived.items():
        if values is None:
            continue
        if col not in df.columns:
            df[col] = np.nan
        values = values.replace([np.inf, -np.inf], np.nan)
        fill = missing_mask(df[col]) & values.notna()
        df.loc[fill, col] = values.loc[fill]
        counts[col] = int(fill.sum())
    return counts
def extract_needed_structure_features(row: pd.Series, pdb_path: str) -> dict[str, Any]:
    """Compute only missing structure features for one row."""
    needed = set(structure_columns_missing(row))
    if not needed:
        return {}
    structure = parse_structure(pdb_path)
    if structure is None:
        return {'_structure_error': 'parse_failed'}
    feats: dict[str, Any] = {}
    plddt_cols = {'avg_plddt', 'min_plddt', 'max_plddt', 'plddt_very_high_pct', 'plddt_high_pct', 'plddt_medium_pct', 'plddt_low_pct'}
    if needed & plddt_cols:
        feats.update(extract_plddt_scores(structure))
    ss_cols = {'helix_percent', 'sheet_percent', 'loop_percent', 'helix_sheet_contrast', 'ordered_percent'}
    if needed & ss_cols:
        feats.update(extract_secondary_structure(structure))
    if 'surface_exposure' in needed:
        feats['surface_exposure'] = calculate_surface_exposure(structure)
    if 'avg_cb_distance' in needed or 'centralization' in needed:
        feats['avg_cb_distance'] = calculate_avg_cb_distance(structure)
    if 'compactness' in needed or 'structural_compactness' in needed:
        feats['compactness'] = calculate_compactness(structure)
    if 'rco' in needed:
        feats['rco'] = calculate_contact_order(structure)
    if 'compactness' in feats and (not is_missing(feats['compactness'])) and (feats['compactness'] > 0):
        feats['structural_compactness'] = 1.0 / feats['compactness']
    if 'avg_cb_distance' in feats and (not is_missing(feats['avg_cb_distance'])) and (feats['avg_cb_distance'] > 0):
        feats['centralization'] = 1.0 / feats['avg_cb_distance']
    return feats
def apply_structure_recovery(df: pd.DataFrame, cache_dirs: list[Path], download_missing: bool, max_rows: int | None) -> tuple[dict[str, int], pd.DataFrame]:
    counts = fill_cheap_structure_derivatives(df)
    structure_map = load_structure_map(cache_dirs)
    status_rows = []
    if 'pdb_path' not in df.columns:
        df['pdb_path'] = np.nan
    path_from_cache = df['Entry'].map(structure_map)
    fill_path = missing_mask(df['pdb_path']) & path_from_cache.notna()
    df.loc[fill_path, 'pdb_path'] = path_from_cache.loc[fill_path]
    counts['pdb_path'] = counts.get('pdb_path', 0) + int(fill_path.sum())
    needs = df[missing_mask(df['pdb_path']) | df.apply(lambda row: bool(structure_columns_missing(row)), axis=1)].copy()
    if max_rows is not None:
        needs = needs.head(max_rows)
    for (idx, row) in tqdm(needs.iterrows(), total=len(needs), desc='Structure recovery'):
        entry = str(row['Entry'])
        path = row.get('pdb_path')
        downloaded = False
        error = ''
        if is_missing(path):
            if download_missing:
                try:
                    path = download_alphafold_structure(entry, str(implement_uniprot_pdb_retrieval_HERE / 'alphafold_cache'))
                    downloaded = bool(path)
                except Exception as exc:
                    path = None
                    error = str(exc)
            if path:
                df.at[idx, 'pdb_path'] = path
                counts['pdb_path'] = counts.get('pdb_path', 0) + 1
        computed_cols: list[str] = []
        if path:
            refreshed = df.loc[idx].copy()
            refreshed['pdb_path'] = path
            try:
                feats = extract_needed_structure_features(refreshed, str(path))
                error = feats.pop('_structure_error', error)
                for (col, val) in feats.items():
                    if col not in df.columns:
                        df[col] = np.nan
                    if col in STRUCTURE_DERIVED_COLUMNS and is_missing(df.at[idx, col]) and (not is_missing(val)):
                        df.at[idx, col] = val
                        counts[col] = counts.get(col, 0) + 1
                        computed_cols.append(col)
            except Exception as exc:
                error = str(exc)
        status_rows.append({'Entry': entry, 'source': row.get('source'), 'had_path_initially': not is_missing(row.get('pdb_path')), 'path': path or '', 'downloaded': downloaded, 'computed_columns': ';'.join(computed_cols), 'error': error})
    return (counts, pd.DataFrame(status_rows))
def write_summary(outdir: Path, input_path: Path, output_path: Path, n_rows: int, uni_raw: pd.DataFrame, uni_failures: pd.DataFrame, fill_counts: dict[str, int], structure_counts: dict[str, int], structure_status: pd.DataFrame) -> None:
    lines = ['# Retrieval Recipe Implementation', '', f'Input: `{input_path}`', f'Output: `{output_path}`', f'Rows: {n_rows:,}', '', '## UniProt Fetch', f'- Returned rows: {len(uni_raw):,}', f'- Missing/not returned: {len(uni_failures):,}', '', '## Filled UniProt/Derived Columns']
    for (col, n) in sorted(fill_counts.items(), key=lambda x: (-x[1], x[0])):
        if n:
            lines.append(f'- {col}: {n:,}')
    lines.extend(['', '## Filled Structure Columns'])
    for (col, n) in sorted(structure_counts.items(), key=lambda x: (-x[1], x[0])):
        if n:
            lines.append(f'- {col}: {n:,}')
    if not structure_status.empty:
        lines.extend(['', '## Structure Recovery Status', f'- Rows attempted: {len(structure_status):,}', f"- Downloaded AlphaFold structures: {int(structure_status['downloaded'].sum()):,}", f"- Rows with errors: {int(structure_status['error'].astype(str).str.len().gt(0).sum()):,}"])
    lines.extend(['', '## Diagnostics', '- `uniprot_fetch.tsv`', '- `uniprot_fetch_failures.csv`', '- `fill_counts.json`', '- `structure_recovery_status.csv`'])
    (outdir / 'retrieval_recipe_report.md').write_text('\n'.join(lines) + '\n')
def implement_uniprot_pdb_retrieval_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument('--uniprot-batch-size', type=int, default=100)
    parser.add_argument('--uniprot-sleep', type=float, default=0.1)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--reuse-uniprot-fetch', type=Path, default=None, help='Optional existing UniProt TSV to reuse instead of fetching.')
    parser.add_argument('--no-download-structures', action='store_true')
    parser.add_argument('--max-structure-rows', type=int, default=None, help='Debug cap for structure recovery rows.')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)
    entries = sorted(df['Entry'].astype(str).unique())
    print(f'Loaded {len(df):,} rows from {args.input}')
    if args.reuse_uniprot_fetch is not None:
        print(f'Reusing UniProt metadata from {args.reuse_uniprot_fetch}')
        uni_raw = pd.read_csv(args.reuse_uniprot_fetch, sep='\t', low_memory=False)
        returned = set(uni_raw['Entry'].astype(str)) if 'Entry' in uni_raw.columns else set()
        uni_failures = pd.DataFrame([{'Entry': entry, 'reason': 'not_in_reused_uniprot_fetch'} for entry in entries if entry not in returned])
    else:
        print(f'Fetching UniProt metadata for {len(entries):,} unique accessions...')
        (uni_raw, uni_failures) = batch_fetch_uniprot(entries, batch_size=args.uniprot_batch_size, sleep_s=args.uniprot_sleep, retries=args.retries)
    if uni_failures.empty:
        uni_failures = pd.DataFrame(columns=['Entry', 'reason'])
    uni_raw.to_csv(args.outdir / 'uniprot_fetch.tsv', sep='\t', index=False)
    uni_failures.to_csv(args.outdir / 'uniprot_fetch_failures.csv', index=False)
    print(f'Preparing UniProt fills ({len(uni_raw):,} fetched rows)...')
    (uni, classifier_df) = prepare_uniprot_for_merge(uni_raw)
    fill_counts = apply_uniprot_fills(df, uni, classifier_df)
    print('Recovering structure paths and missing structure-derived columns...')
    cache_dirs = [implement_uniprot_pdb_retrieval_ROOT / 'data' / 'pdbs_pifold_downloaded', implement_uniprot_pdb_retrieval_HERE / 'alphafold_cache', implement_uniprot_pdb_retrieval_HERE / 'pdb_cache']
    (structure_counts, structure_status) = apply_structure_recovery(df, cache_dirs=cache_dirs, download_missing=not args.no_download_structures, max_rows=args.max_structure_rows)
    structure_status.to_csv(args.outdir / 'structure_recovery_status.csv', index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    (args.outdir / 'fill_counts.json').write_text(json.dumps({'uniprot_and_derived': fill_counts, 'structure': structure_counts}, indent=2, sort_keys=True))
    write_summary(args.outdir, args.input, args.output, len(df), uni_raw, uni_failures, fill_counts, structure_counts, structure_status)
    print(f'Wrote enriched CSV: {args.output}')
    print(f'Wrote diagnostics: {args.outdir}')
def implement_uniprot_pdb_retrieval__entry():
    sys.path.insert(0, str(implement_uniprot_pdb_retrieval_ROOT))
    implement_uniprot_pdb_retrieval_main()

# ---------- from prepare_pdb_chain_sequences.py ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_FINAL.csv'
OUTPUT = REPO_ROOT / 'dataset_update' / 'expansion_for_scoring_monomer_PDB_FINAL.csv'
PDB_CACHE = REPO_ROOT / 'dataset_update' / 'pdb_cache'
IDENTITY_THRESHOLD = 0.95
ALLOW_X_IN_CHAIN = False
THREE_TO_ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
MODIFIED_TO_PARENT = {'MSE': 'M', 'SEP': 'S', 'TPO': 'T', 'PTR': 'Y', 'HYP': 'P', 'PCA': 'E', 'CME': 'C', 'CSO': 'C', 'KCX': 'K', 'MLY': 'K', 'LLP': 'K', 'CSD': 'C', 'OCS': 'C', 'CAS': 'C'}
def download_pdb(pdb_id: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 100:
        return True
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
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
    seen = {}
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
    for ((rn, ic), code) in chain_residues:
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
        return (0, 0.0)
    if n_chain > n_uni:
        (best_offset, best_id) = (0, 0.0)
        for offset in range(n_chain - n_uni + 1):
            window = chain_seq[offset:offset + n_uni]
            ident = sum((1 for (x, y) in zip(window, uniprot_seq) if x == y)) / n_uni
            if ident > best_id:
                best_id = ident
                best_offset = -offset
        return (best_offset, best_id)
    (best_offset, best_id) = (0, 0.0)
    for offset in range(n_uni - n_chain + 1):
        window = uniprot_seq[offset:offset + n_chain]
        ident = sum((1 for (x, y) in zip(window, chain_seq) if x == y)) / n_chain
        if ident > best_id:
            best_id = ident
            best_offset = offset
    return (best_offset, best_id)
def process_one(row: pd.Series) -> dict:
    """Worker: returns dict with chain sequence + alignment info."""
    pdb_id = row['pdb_id']
    chain = row['pdb_chain']
    uniprot_seq = row['sequence']
    pdb_path = PDB_CACHE / f'{pdb_id}.pdb'
    if not download_pdb(pdb_id, pdb_path):
        return {'Entry': row['Entry'], 'pdb_chain_sequence': None, 'pdb_chain_offset': None, 'pdb_chain_identity': 0.0, 'len_chain': 0, 'fail_reason': 'download_failed'}
    chain_seq = parse_chain_sequence(pdb_path, chain)
    if not chain_seq:
        return {'Entry': row['Entry'], 'pdb_chain_sequence': None, 'pdb_chain_offset': None, 'pdb_chain_identity': 0.0, 'len_chain': 0, 'fail_reason': 'chain_not_found'}
    (offset, identity) = best_alignment(uniprot_seq, chain_seq)
    has_x = 'X' in chain_seq
    return {'Entry': row['Entry'], 'pdb_chain_sequence': chain_seq, 'pdb_chain_offset': offset, 'pdb_chain_identity': round(identity, 4), 'len_chain': len(chain_seq), 'has_x_in_chain': has_x, 'fail_reason': None}
def prepare_pdb_chain_sequences_main():
    PDB_CACHE.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT, low_memory=False)
    print(f'Loaded {len(df):,} entries from {INPUT.name}')
    pdb_df = df[df['pdb_available']].reset_index(drop=True)
    print(f'  with pdb_available=True: {len(pdb_df):,}')
    results = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(process_one, row): i for (i, row) in pdb_df.iterrows()}
        for f in tqdm(as_completed(futures), total=len(futures), desc='Extracting chain seqs'):
            results.append(f.result())
    info = pd.DataFrame(results)
    merged = pdb_df.merge(info, on='Entry', how='left')
    print(f'\n=== Pre-filter stats (n={len(merged):,}) ===')
    print(f"  Identity == 1.000:        {(merged['pdb_chain_identity'] >= 0.999).sum():>5,d}")
    print(f"  Identity ≥ 0.95:          {(merged['pdb_chain_identity'] >= 0.95).sum():>5,d}")
    print(f"  Identity ≥ 0.90:          {(merged['pdb_chain_identity'] >= 0.9).sum():>5,d}")
    print(f"  Identity < 0.90:          {(merged['pdb_chain_identity'] < 0.9).sum():>5,d}")
    print(f"  Offset == 0 (no shift):   {(merged['pdb_chain_offset'] == 0).sum():>5,d}")
    print(f"  Offset ≠ 0 (shifted):     {(merged['pdb_chain_offset'] != 0).sum():>5,d}")
    print(f"  Has 'X' in chain:         {merged['has_x_in_chain'].sum():>5,d}")
    print(f"  Download/chain failures:  {merged['fail_reason'].notna().sum():>5,d}")
    print(f'\n  Offset distribution (top 10):')
    print(merged['pdb_chain_offset'].value_counts().head(10).to_string())
    print(f'\n=== Filtering ===')
    keep = merged['pdb_chain_identity'] >= IDENTITY_THRESHOLD
    if not ALLOW_X_IN_CHAIN:
        keep = keep & ~merged['has_x_in_chain'].fillna(True)
    keep = keep & merged['fail_reason'].isna()
    final = merged[keep].copy()
    final['uniprot_sequence_length'] = final['sequence'].str.len()
    final['sequence'] = final['pdb_chain_sequence']
    final = final.drop(columns=['has_x_in_chain', 'fail_reason'])
    print(f'  Kept: {len(final):,} / {len(merged):,} ({100 * len(final) / len(merged):.1f}%)')
    print(f'\n=== Final dataset domain distribution ===')
    print(final['domain'].value_counts().to_string())
    final.to_csv(OUTPUT, index=False)
    print(f'\nSaved {len(final):,} entries to {OUTPUT.name}')
    print(f'  Path: {OUTPUT}')
def prepare_pdb_chain_sequences__entry():
    prepare_pdb_chain_sequences_main()

_STEPS = {
    'build-independent-pdb-cohort': build_independent_pdb_cohort__entry,
    'build-mif-safe-cohort': build_mif_safe_cohort__entry,
    'scan-uniprot-pdb': scan_uniprot_pdb__entry,
    'implement-uniprot-pdb-retrieval': implement_uniprot_pdb_retrieval__entry,
    'prepare-pdb-chain-sequences': prepare_pdb_chain_sequences__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

