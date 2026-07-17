"""decoding_bias.finetune.mpnn_data -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - build_dataset
  - build_alkaline_dataset
  - assign_structures_alkaliphile
  - extract_alkaline_structures
  - extract_alkaliphile_structures
  - stage_c_chain_qc
  - stage_d_alkaliphile
  - stage_d_cluster_split
  - prep_secreted_targets
"""

import argparse
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio
import biotite.structure.io.pdbx as pdbx
import collections
import csv
import extract_alkaline_structures as E
import io
import json
import numpy as np
import pandas as pd
import random
import re
import requests
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed
from _cohort import cfg
from build_pdb_scoring_inputs import three_to_one
from Bio import Align
from surface_features_alkaline import per_residue_rsa, RSA_CUT

# ---------- from build_dataset.py ----------
build_dataset_HERE = Path(__file__).resolve().parent
build_dataset_OUT = build_dataset_HERE / 'outputs'
PER_ORG_CAP = 150
(LEN_MIN, LEN_MAX) = (40, 700)
_STD = set('ACDEFGHIKLMNPQRSTVWY')
def gravy_of(seq):
    """Kyte-Doolittle GRAVY of a sequence (standard residues only); used to match controls on
    hydrophobicity so the case-control contrast isolates surface CHARGE, not bulk hydrophobicity."""
    c = ''.join((x for x in str(seq) if x in _STD))
    try:
        return ProteinAnalysis(c).gravy() if c else 0.0
    except Exception:
        return 0.0
PANELS = {'alkaline': {'label': 'alkaliphile', 'tight_family': 'Bacillaceae', 'cases': {79880: ('A. clausii', 'Bacilli'), 79681: ('A. pseudofirmus', 'Bacilli'), 86665: ('B. halodurans', 'Bacilli'), 1445: ('A. alcalophilus', 'Bacilli'), 1218: ('Alkaliphilus metalliredigens', 'Clostridia'), 461876: ('Alkaliphilus oremlandii', 'Clostridia'), 375929: ('Natranaerobius thermophilus', 'Clostridia'), 490314: ('Dethiobacter alkaliphilus', 'Clostridia'), 106633: ('Thioalkalivibrio', 'Gammaproteobacteria'), 2257: ('Natronomonas pharaonis', 'Haloarchaea'), 29288: ('Natronococcus occultus', 'Haloarchaea'), 13769: ('Natrialba magadii', 'Haloarchaea'), 44930: ('Natronobacterium gregoryi', 'Haloarchaea')}, 'neutral': {1423: ('B. subtilis', 'Bacilli'), 1402: ('B. licheniformis', 'Bacilli'), 1390: ('B. amyloliquefaciens', 'Bacilli'), 1488: ('Clostridium acetobutylicum', 'Clostridia'), 1502: ('Clostridium perfringens', 'Clostridia'), 562: ('E. coli', 'Gammaproteobacteria'), 287: ('Pseudomonas aeruginosa', 'Gammaproteobacteria'), 2746: ('Halomonas elongata', 'Gammaproteobacteria'), 2242: ('Halobacterium salinarum', 'Haloarchaea'), 2246: ('Haloferax volcanii', 'Haloarchaea')}}, 'acid': {'label': 'acidophile', 'tight_family': 'Acidithiobacillus', 'cases': {920: ('A. ferrooxidans', 'Gammaproteobacteria'), 930: ('A. thiooxidans', 'Gammaproteobacteria'), 33059: ('A. caldus', 'Gammaproteobacteria'), 524: ('Acidiphilium cryptum', 'Alphaproteobacteria'), 179: ('Leptospirillum', 'Nitrospira')}, 'neutral': {562: ('E. coli', 'Gammaproteobacteria'), 287: ('Pseudomonas aeruginosa', 'Gammaproteobacteria'), 294: ('Pseudomonas fluorescens', 'Gammaproteobacteria'), 90371: ('Salmonella Typhimurium', 'Gammaproteobacteria'), 155892: ('Caulobacter crescentus', 'Alphaproteobacteria')}}}
def harvest_secreted(taxid):
    url = 'https://rest.uniprot.org/uniprotkb/search'
    q = f'(taxonomy_id:{taxid}) AND (cc_scl_term:SL-0243 OR ft_signal:*) AND (length:[{LEN_MIN} TO {LEN_MAX}])'
    params = {'query': q, 'format': 'tsv', 'size': 500, 'fields': 'accession,length,sequence,organism_name,lineage,xref_pfam,ec,protein_name,ft_signal,cc_subcellular_location'}
    frames = []
    while True:
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        frames.append(pd.read_csv(io.StringIO(r.text), sep='\t'))
        nxt = r.links.get('next', {}).get('url')
        if not nxt:
            break
        (url, params) = (nxt, None)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
def build_dataset_pfam_set(s):
    return set(re.findall('(PF\\d{5})', s)) if isinstance(s, str) else set()
def build_dataset_ec_parts(s):
    if not isinstance(s, str) or not s.strip():
        return ('', '', '')
    e = s.split(';')[0].strip()
    p = e.split('.')
    ec3 = '.'.join(p[:3]) if len(p) >= 3 and '-' not in p[:3] else ''
    ec2 = '.'.join(p[:2]) if len(p) >= 2 and '-' not in p[:2] else ''
    return (e, ec3, ec2)
def build_dataset_collect(panel, group):
    rows = []
    for (tid, (name, clade)) in panel.items():
        d = harvest_secreted(tid)
        if len(d):
            d = d.drop_duplicates('Entry')
            if len(d) > PER_ORG_CAP:
                d = d.sample(PER_ORG_CAP, random_state=0)
            d['taxid'] = tid
            d['org_short'] = name
            d['clade'] = clade
            d['group'] = group
            rows.append(d)
        print(f'  {group:<13} {tid:>7} {name:<28} secreted(40-700): {(0 if not len(d) else len(d))}', flush=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df.drop_duplicates('Sequence') if 'Sequence' in df.columns else df
def prep(df, tight_family):
    df = df.rename(columns={'Entry': 'acc', 'Length': 'length', 'Sequence': 'sequence', 'Organism': 'organism', 'Taxonomic lineage': 'lineage', 'Pfam': 'pfam_raw', 'EC number': 'ec_raw', 'Protein names': 'protein_name'})
    df['pfam'] = df.pfam_raw.apply(lambda s: ';'.join(sorted(build_dataset_pfam_set(s))))
    ecp = df.ec_raw.apply(build_dataset_ec_parts)
    df['ec_full'] = [a for (a, _, _) in ecp]
    df['ec3'] = [b for (_, b, _) in ecp]
    df['ec2'] = [c for (_, _, c) in ecp]
    df['ec_class'] = df.ec_full.apply(lambda e: e.split('.')[0] if isinstance(e, str) and e else '')
    df['gravy'] = df.sequence.apply(gravy_of)
    df['in_tight_subset'] = df.lineage.apply(lambda s: isinstance(s, str) and tight_family in s)
    df['secreted_evidence'] = df.apply(lambda r: 'SCL_secreted' if isinstance(r.get('Subcellular location [CC]'), str) and 'Secreted' in r['Subcellular location [CC]'] else 'signal_peptide', axis=1)
    return df
def build_dataset_match(cases, pool, gravy_band=None):
    used = set()
    res = {}
    for case in sorted(cases, key=lambda c: (c['length'] or 0, c['acc'])):
        cpf = build_dataset_pfam_set(case['pfam'])
        base = [c for c in pool if c['acc'] not in used and case['length'] and c['length'] and (abs(c['length'] - case['length']) <= 0.25 * case['length'])]
        tiers = [('Pfam', lambda c: bool(cpf & build_dataset_pfam_set(c['pfam']))), ('EC3', lambda c: case['ec3'] and c['ec3'] == case['ec3']), ('EC2', lambda c: case['ec2'] and c['ec2'] == case['ec2']), ('EC_class', lambda c: case['ec_class'] and c['ec_class'] == case['ec_class'])]
        (chosen, tier) = (None, 'unmatched')
        for (tname, pred) in tiers:
            cand = [c for c in base if pred(c)]
            if not cand:
                continue
            if gravy_band is not None:
                gd = lambda c: abs(c.get('gravy', 0.0) - case.get('gravy', 0.0))
                near = [c for c in cand if gd(c) <= gravy_band]
                if near:
                    cand = near
                key = lambda c: (gd(c), c['clade'] == case['clade'], abs(c['length'] - case['length']), c['acc'])
            else:
                key = lambda c: (c['clade'] == case['clade'], abs(c['length'] - case['length']), c['acc'])
            chosen = min(cand, key=key)
            tier = tname
            used.add(chosen['acc'])
            break
        res[case['acc']] = (tier, chosen)
    return res
def build_dataset_main(cohort):
    P = PANELS[cohort]
    lab = P['label']
    print(f'[cohort={cohort}] harvesting secreted proteins ({lab}) ...')
    cas = prep(build_dataset_collect(P['cases'], lab), P['tight_family'])
    print(f'[cohort={cohort}] harvesting secreted proteins (neutralophiles) ...')
    neu = prep(build_dataset_collect(P['neutral'], 'neutralophile'), P['tight_family'])
    print(f'\n{lab} cases {len(cas)} | neutralophile pool {len(neu)}')
    print(f'  {lab} by clade:', dict(cas.clade.value_counts()))
    cases = cas.to_dict('records')
    pool = neu.to_dict('records')
    gravy_band = P.get('gravy_band')
    m = build_dataset_match(cases, pool, gravy_band)
    npair = sum((1 for c in cases if m[c['acc']][1]))
    cross = sum((1 for c in cases if m[c['acc']][1] and m[c['acc']][1]['clade'] != c['clade']))
    tiers = collections.Counter((m[c['acc']][0] for c in cases if m[c['acc']][1]))
    print(f'\nmatched pairs {npair}/{len(cases)} | cross-clade {cross} | tiers {dict(tiers)}')
    if gravy_band is not None:
        dg = [m[c['acc']][1]['gravy'] - c['gravy'] for c in cases if m[c['acc']][1]]
        mc = sum((c['gravy'] for c in cases if m[c['acc']][1])) / max(1, npair)
        mk = mc + sum(dg) / max(1, npair)
        print(f'  GRAVY match (band ±{gravy_band}): case {mc:+.3f} vs control {mk:+.3f} (mean case-control gap {-sum(dg) / max(1, npair):+.3f}; was +0.138 unmatched)')
    keep = ['acc', 'group', 'clade', 'org_short', 'organism', 'in_tight_subset', 'secreted_evidence', 'pfam', 'ec_full', 'ec3', 'ec_class', 'length', 'gravy', 'sequence']
    (crows, ctrows) = ([], [])
    for c in cases:
        (t, ctrl) = m[c['acc']]
        row = {k: c.get(k, '') for k in keep}
        row.update(match_tier=t, matched_control_uniprot=ctrl['acc'] if ctrl else '', cross_clade=ctrl['clade'] != c['clade'] if ctrl else '')
        crows.append(row)
        if ctrl:
            cr = {k: ctrl.get(k, '') for k in keep}
            cr.update(matched_case_uniprot=c['acc'], match_tier=t, cross_clade=ctrl['clade'] != c['clade'])
            ctrows.append(cr)
    build_dataset_OUT.mkdir(exist_ok=True)
    pd.DataFrame(crows).to_csv(build_dataset_OUT / f'{lab}_cases_stageC.csv', index=False)
    pd.DataFrame(ctrows).to_csv(build_dataset_OUT / f'{lab}_neutral_controls_stageC.csv', index=False)
    print(f'\nwrote {lab}_cases_stageC.csv ({len(crows)}) + {lab}_neutral_controls_stageC.csv ({len(ctrows)})')
    arrow = 'ACIDIC (net negative)' if cohort == 'alkaline' else 'BASIC (net positive, the reverse)'
    print(f'EXPECTED on fine-tuning: designs go {arrow}.')
def build_dataset__entry():
    warnings.filterwarnings('ignore')
    random.seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True, choices=['alkaline', 'acid'])
    build_dataset_main(ap.parse_args().cohort)

# ---------- from build_alkaline_dataset.py ----------
build_alkaline_dataset_HERE = Path(__file__).resolve().parent
build_alkaline_dataset_OUT = build_alkaline_dataset_HERE / 'outputs'
B = 'https://rest.uniprot.org/uniprotkb/search'
AFDB = 'https://alphafold.ebi.ac.uk/api/prediction/'
DOMAINS = {'Bacteria', 'Archaea', 'Eukaryota'}
(build_alkaline_dataset_COV_PASS, build_alkaline_dataset_COV_BORDER, build_alkaline_dataset_RES_XRAY, build_alkaline_dataset_RES_EM, LEN_MIN, LEN_MAX) = (0.8, 0.7, 3.0, 4.0, 40, 700)
(AF_PLDDT, AF_CONF, AF_VLOW) = (70.0, 0.7, 0.3)
def ph_opt(text):
    m = re.search('optimum ph(?:\\s*is|:)?\\s*(?:around|about|approximately|~|>|>=|of)?\\s*([\\d]+(?:\\.\\d+)?)(?:\\s*(?:-|-|to|and)\\s*([\\d]+(?:\\.\\d+)?))?', text.lower())
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    return (min(a, b), max(a, b))
def harvest(query):
    rows = []
    url = B
    params = {'query': query, 'format': 'json', 'size': 500}
    while url:
        r = requests.get(url, params=params, timeout=120)
        params = None
        if r.status_code != 200:
            raise RuntimeError(f'{r.status_code}: {r.text[:200]}')
        for e in r.json().get('results', []):
            ph = None
            ev = []
            scl = []
            subunit = ''
            for c in e.get('comments', []):
                if c.get('phDependence'):
                    txts = c['phDependence'].get('texts', [])
                    ph = ' '.join((t['value'] for t in txts))
                    ev = [f"PubMed:{x['id']}" for t in txts for x in t.get('evidences', []) if x.get('source') == 'PubMed']
                if c.get('commentType') == 'SUBCELLULAR LOCATION':
                    scl += [l.get('location', {}).get('value', '') for l in c.get('subcellularLocations', [])]
                if c.get('commentType') == 'SUBUNIT':
                    subunit = ' '.join((t['value'] for t in c.get('texts', [])))[:60]
            opt = ph_opt(ph) if ph else None
            pd_ = e.get('proteinDescription', {}).get('recommendedName', {})
            ec = ';'.join((x['value'] for x in pd_.get('ecNumbers', []))) if pd_.get('ecNumbers') else ''
            kws = [k.get('name') for k in e.get('keywords', [])]
            pfam = ';'.join((x['id'] for x in e.get('uniProtKBCrossReferences', []) if x['database'] == 'Pfam'))
            pdb = [(x['id'], {p['key']: p['value'] for p in x.get('properties', [])}) for x in e.get('uniProtKBCrossReferences', []) if x['database'] == 'PDB']
            lineage = e.get('organism', {}).get('lineage', [])
            dom = next((d for d in DOMAINS if d in lineage), lineage[0] if lineage else '?')
            rows.append(dict(acc=e['primaryAccession'], length=e.get('sequence', {}).get('length'), sequence=e.get('sequence', {}).get('value', ''), opt_lo=opt[0] if opt else None, opt_hi=opt[1] if opt else None, ph_text=(ph or '').replace('\n', ' ')[:220], evidence_ids=';'.join(ev), ec_full=ec, secreted='Secreted' in kws or any(('Secreted' in s for s in scl)), localization=';'.join(sorted(set(scl)))[:80], oligomeric_note=subunit, pfam=pfam, domain=dom, organism=e.get('organism', {}).get('scientificName', ''), pdb=pdb))
        m = re.search('<([^>]+)>;\\s*rel="next"', r.headers.get('link', ''))
        url = m.group(1) if m else None
    return rows
def build_alkaline_dataset_ec_parts(ec):
    e = (ec or '').split(';')[0].strip()
    p = e.split('.') if e and e[0].isdigit() else []
    return ('.'.join(p[:3]) if len(p) >= 3 and '-' not in p[2] else '', '.'.join(p[:2]) if len(p) >= 2 and '-' not in p[1] else '', p[0] if p else 'none')
def best_pdb(pdb, ulen):
    best = None
    for (pid, props) in pdb:
        method = props.get('Method', '')
        res = None
        mr = re.search('([\\d.]+)\\s*A', props.get('Resolution', '') or '')
        if mr:
            res = float(mr.group(1))
        spans = re.findall('=(\\d+)-(\\d+)', props.get('Chains', '') or '')
        if not spans or not ulen:
            continue
        st = min((int(a) for (a, _) in spans))
        en = max((int(b) for (_, b) in spans))
        chain = props.get('Chains', '').split('=')[0].split('/')[0]
        cov = (en - st + 1) / ulen
        score = ({'X-ray': 0, 'EM': 1, 'NMR': 2}.get(method, 3), -cov, res if res is not None else 9.9)
        rec = dict(pdb_id=pid, chain=chain, method=method, res=res, start=st, end=en, cov=round(cov, 3))
        if best is None or score < best[0]:
            best = (score, rec)
    return best[1] if best else None
def pdb_qc(rec):
    if rec is None:
        return (False, 'no_chain', False)
    f = []
    if rec['method'] == 'X-ray' and (rec['res'] is None or rec['res'] > build_alkaline_dataset_RES_XRAY):
        f.append('res>3.0')
    elif rec['method'] == 'EM' and (rec['res'] is None or rec['res'] > build_alkaline_dataset_RES_EM):
        f.append('EM_res>4.0')
    elif rec['method'] not in ('X-ray', 'EM', 'NMR'):
        f.append(f"method={rec['method'] or '?'}")
    if rec['cov'] < build_alkaline_dataset_COV_BORDER:
        f.append('coverage<0.70')
    elif rec['cov'] < build_alkaline_dataset_COV_PASS:
        f.append('coverage_borderline')
    if not LEN_MIN <= rec['end'] - rec['start'] + 1 <= LEN_MAX:
        f.append('length_out_of_range')
    hard = [x for x in f if x != 'coverage_borderline']
    return (len(hard) == 0, ';'.join(f), 'coverage_borderline' in f)
def af_qc(acc):
    try:
        r = requests.get(AFDB + acc, timeout=30)
        if r.status_code != 200:
            return None
        d = r.json()[0]
        mean = d.get('globalMetricValue')
        conf = d.get('fractionPlddtConfident', 0) + d.get('fractionPlddtVeryHigh', 0)
        vlow = d.get('fractionPlddtVeryLow', 0)
        ok = mean is not None and mean >= AF_PLDDT and (conf >= AF_CONF) and (vlow <= AF_VLOW)
        return dict(ok=ok, mean=round(mean, 1) if mean else None, conf=round(conf, 3), vlow=round(vlow, 3), url=d.get('pdbUrl', ''))
    except Exception:
        return None
def assign_structure(rows):
    """Set structure_source + QC for each row in-place. PDB-first, AF fallback (threaded)."""
    need_af = []
    for r in rows:
        rec = best_pdb(r['pdb'], r['length'])
        (ok, flags, border) = pdb_qc(rec)
        r['_pdb_rec'] = rec
        if ok:
            r.update(structure_source='PDB', pdb_id=rec['pdb_id'], pdb_chain=rec['chain'], method=rec['method'], resolution=rec['res'], uniprot_start=rec['start'], uniprot_end=rec['end'], chain_coverage_fraction=rec['cov'], af_mean_plddt='', af_conf_fraction='', af_verylow_fraction='', qc_pass=True, qc_flags='coverage_borderline' if border else '', _border=border)
        else:
            need_af.append(r)

    def work(r):
        return (r, af_qc(r['acc']))
    with ThreadPoolExecutor(max_workers=12) as ex:
        for (r, af) in (f.result() for f in as_completed([ex.submit(work, r) for r in need_af])):
            len_ok = bool(r['length']) and LEN_MIN <= r['length'] <= LEN_MAX
            if af and af['ok'] and len_ok:
                r.update(structure_source='AF', pdb_id='', pdb_chain='', method='AlphaFold', resolution='', uniprot_start='', uniprot_end='', chain_coverage_fraction='', af_mean_plddt=af['mean'], af_conf_fraction=af['conf'], af_verylow_fraction=af['vlow'], qc_pass=True, qc_flags='', _border=False)
            else:
                why = 'af_length_out_of_range' if af and af['ok'] and (not len_ok) else 'pdb_fail+af_fail' if af else 'pdb_fail+no_af'
                r.update(structure_source='none', qc_pass=False, qc_flags=why, _border=False, pdb_id='', pdb_chain='', method='', resolution='', uniprot_start='', uniprot_end='', chain_coverage_fraction='', af_mean_plddt=af['mean'] if af else '', af_conf_fraction=af['conf'] if af else '', af_verylow_fraction=af['vlow'] if af else '')
def build_alkaline_dataset_pfam_set(s):
    return set((x for x in (s or '').split(';') if x))
def build_alkaline_dataset_match(cases, pool):
    used = set()
    res = {}
    pool_ok = [c for c in pool if c['qc_pass']]
    for case in sorted(cases, key=lambda c: (c['length'] or 0, c['acc'])):
        chosen = None
        tier = 'unmatched'
        cpf = build_alkaline_dataset_pfam_set(case['pfam'])
        base = [c for c in pool_ok if c['acc'] not in used and c['domain'] == case['domain'] and case['length'] and c['length'] and (abs(c['length'] - case['length']) <= 0.25 * case['length'])]
        tiers = [('Pfam', lambda c: bool(cpf & build_alkaline_dataset_pfam_set(c['pfam']))), ('EC3', lambda c: case['ec3'] and c['ec3'] == case['ec3']), ('EC2', lambda c: case['ec2'] and c['ec2'] == case['ec2']), ('EC_class', lambda c: c['ec_class'] == case['ec_class'])]
        for (tname, pred) in tiers:
            cand = [c for c in base if pred(c)]
            if cand:
                chosen = min(cand, key=lambda c: (c['structure_source'] != case['structure_source'], abs(c['length'] - case['length']), c['acc']))
                tier = tname
                used.add(chosen['acc'])
                break
        res[case['acc']] = (tier, chosen)
    return res
STRUCT_COLS = ['structure_source', 'pdb_id', 'pdb_chain', 'method', 'resolution', 'uniprot_start', 'uniprot_end', 'chain_coverage_fraction', 'af_mean_plddt', 'af_conf_fraction', 'af_verylow_fraction', 'qc_pass', 'qc_flags']
def write_set(tag, cases, matches, conf):
    base = ['acc', 'label_source', 'case_set', 'pH_optimum_value', 'pH_optimum_range', 'range_width', 'label_confidence', 'annotation_text', 'evidence_ids', 'ec_full', 'ec3', 'ec_class', 'pfam', 'domain', 'organism', 'localization', 'secreted', 'oligomeric_note', 'length', 'sequence', 'surface_charge_features_available']
    extra = ['match_tier', 'matched_control_uniprot', 'cluster_id', 'sequence_identity_to_control', 'split']
    with open(build_alkaline_dataset_OUT / f'alkaline_optimum_cases_{tag}_stageC.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=base + STRUCT_COLS + extra, extrasaction='ignore')
        w.writeheader()
        for r in cases:
            (t, c) = matches[r['acc']]
            row = {k: r.get(k, '') for k in base + STRUCT_COLS}
            row.update(label_source='UniProt pH optimum', case_set=tag, pH_optimum_value=round((r['opt_lo'] + r['opt_hi']) / 2, 2), pH_optimum_range=f"{r['opt_lo']}-{r['opt_hi']}", range_width=round(r['opt_hi'] - r['opt_lo'], 2), label_confidence=conf(r), annotation_text=r['ph_text'], surface_charge_features_available=True, match_tier=t, matched_control_uniprot=c['acc'] if c else '', cluster_id='TBD', sequence_identity_to_control='TBD', split='TBD')
            w.writerow(row)
    cbase = ['acc', 'label_source', 'for_case_set', 'pH_optimum_value', 'pH_optimum_range', 'ec_full', 'ec3', 'ec_class', 'pfam', 'domain', 'organism', 'length', 'sequence']
    with open(build_alkaline_dataset_OUT / f'matched_neutral_controls_for_{tag}_cases_stageC.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cbase + STRUCT_COLS + ['matched_case_uniprot', 'match_tier', 'cluster_id', 'split'], extrasaction='ignore')
        w.writeheader()
        for r in cases:
            (t, c) = matches[r['acc']]
            if not c:
                continue
            row = {k: c.get(k, '') for k in cbase + STRUCT_COLS}
            row.update(label_source='UniProt pH optimum (neutral)', for_case_set=tag, pH_optimum_value=round((c['opt_lo'] + c['opt_hi']) / 2, 2), pH_optimum_range=f"{c['opt_lo']}-{c['opt_hi']}", matched_case_uniprot=r['acc'], match_tier=t, cluster_id='TBD', split='TBD')
            w.writerow(row)
def build_alkaline_dataset_main():
    print('harvesting ALL reviewed pH-annotated entries (no PDB filter) ...')
    rows = harvest('(cc_bpcp_ph_dependence:*) AND reviewed:true')
    for r in rows:
        (r['ec3'], r['ec2'], r['ec_class']) = build_alkaline_dataset_ec_parts(r['ec_full'])
    parsed = [r for r in rows if r['opt_hi'] is not None]

    def conf(r):
        return 'high' if r['opt_lo'] >= 8.5 and r['opt_hi'] - r['opt_lo'] <= 1.0 else 'medium'
    cases_incl = [r for r in parsed if r['opt_hi'] >= 8.5]
    cases_high = [r for r in cases_incl if r['opt_lo'] >= 8.5 and r['opt_hi'] - r['opt_lo'] <= 1.0]
    pool = [r for r in parsed if r['opt_lo'] >= 6.0 and r['opt_hi'] <= 7.5]
    print(f'parsed {len(parsed)} | high-conf {len(cases_high)} | inclusive {len(cases_incl)} | neutral pool {len(pool)}')
    universe = {r['acc']: r for r in cases_incl + pool}.values()
    print(f'assigning structure_source + per-source QC to {len(list(universe))} proteins (PDB then AF) ...', flush=True)
    universe = list({r['acc']: r for r in cases_incl + pool}.values())
    assign_structure(universe)
    src = collections.Counter((r['structure_source'] for r in universe))
    print(f'  structure_source over union: {dict(src)}')
    for (tag, cs) in [('high_confidence', cases_high), ('inclusive', cases_incl)]:
        cs_ok = [r for r in cs if r['qc_pass']]
        m = build_alkaline_dataset_match(cs_ok, pool)
        write_set(tag, cs_ok, m, conf)
        npair = sum((1 for r in cs_ok if m[r['acc']][1]))
        bs = collections.Counter((r['structure_source'] for r in cs_ok))
        print(f'  [{tag}] QC-pass cases {len(cs_ok)}/{len(cs)} {dict(bs)} | matched pairs {npair}')
    genera = ['Alkalihalobacillus', 'Bacillus halodurans', 'Bacillus pseudofirmus', 'Halalkalibacterium', 'Natronomonas', 'Halorhodospira', 'Alkaliphilus', 'Thioalkalivibrio', 'Natranaerobius', 'Natrialba']
    orgq = ' OR '.join((f'organism_name:"{g}"' for g in genera))
    eco = harvest(f'reviewed:true AND (keyword:KW-0964) AND ({orgq})')
    with open(build_alkaline_dataset_OUT / 'alkaline_ecological_supplement.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['acc', 'organism', 'domain', 'localization', 'ec_full', 'pfam', 'length', 'ph_text', 'sequence'], extrasaction='ignore')
        w.writeheader()
        [w.writerow(r) for r in eco]
    print(f'ecological supplement: {len(eco)} (separate). Next: Stage D clustering/splits on the _stageC.csv.')
def build_alkaline_dataset__entry():
    build_alkaline_dataset_main()

# ---------- from assign_structures_alkaliphile.py ----------
assign_structures_alkaliphile_HERE = Path(__file__).resolve().parent
assign_structures_alkaliphile_OUT = assign_structures_alkaliphile_HERE / 'outputs'
def fetch_pdb_len(accs):
    url = 'https://rest.uniprot.org/uniprotkb/search'
    out = {}
    for i in range(0, len(accs), 80):
        chunk = accs[i:i + 80]
        params = {'query': 'accession:(' + ' OR '.join(chunk) + ')', 'format': 'json', 'size': 100, 'fields': 'accession,length,xref_pdb'}
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=90)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        for e in r.json().get('results', []):
            pdb = [(x['id'], {p['key']: p['value'] for p in x.get('properties', [])}) for x in e.get('uniProtKBCrossReferences', []) if x['database'] == 'PDB']
            out[e['primaryAccession']] = dict(length=e.get('sequence', {}).get('length'), pdb=pdb)
        print(f'  fetched {min(i + 80, len(accs))}/{len(accs)}', flush=True)
    return out
def assign_structures_alkaliphile_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='alkaline', choices=['alkaline', 'acid'])
    a = ap.parse_args()
    C = cfg(a.cohort)
    CASES = assign_structures_alkaliphile_OUT / C['cases_D']
    CTRLS = assign_structures_alkaliphile_OUT / C['ctrls_D']
    ca = pd.read_csv(CASES)
    co = pd.read_csv(CTRLS)
    accs = sorted(set(ca.acc) | set(co.acc))
    print(f'fetching PDB xrefs + length for {len(accs)} accessions ...')
    info = fetch_pdb_len(accs)
    rows = [dict(acc=a, length=info.get(a, {}).get('length'), pdb=info.get(a, {}).get('pdb', [])) for a in accs]
    print('assigning structure_source + per-source QC (PDB then AF) ...', flush=True)
    assign_structure(rows)
    bymap = {r['acc']: r for r in rows}

    def merge(df):
        for col in STRUCT_COLS:
            df[col] = df.acc.map(lambda a: bymap.get(a, {}).get(col, ''))
        return df
    merge(ca).to_csv(CASES, index=False)
    merge(co).to_csv(CTRLS, index=False)
    import collections
    src = collections.Counter((r.get('structure_source', 'none') for r in rows))
    print(f'\nstructure_source over {len(rows)}: {dict(src)}')
    for (name, df) in [('cases', ca), ('controls', co)]:
        sp = df[df.split.notna()]
        npass = int(sp.qc_pass.sum()) if 'qc_pass' in sp else 0
        print(f'  {name}: {npass}/{len(sp)} QC-pass (split-assigned)')
    cok = set(ca[ca.qc_pass == True].acc)
    ook = set(co[co.qc_pass == True].acc)
    pairs = ca[(ca.qc_pass == True) & ca.matched_control_uniprot.isin(ook) & ca.split.notna()]
    print(f'  both-pass matched pairs: {len(pairs)}')
def assign_structures_alkaliphile__entry():
    sys.path.insert(0, str(assign_structures_alkaliphile_HERE))
    assign_structures_alkaliphile_main()

# ---------- from extract_alkaline_structures.py ----------
extract_alkaline_structures_HERE = Path(__file__).resolve().parent
_aligner = Align.PairwiseAligner(mode='global', match_score=1, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5)
def gapped_identity(chain, uni):
    if not chain or not uni:
        return 0.0
    aln = _aligner.align(chain, uni)[0]
    ident = 0
    for ((s1, e1), (s2, e2)) in zip(*aln.aligned):
        ident += sum((1 for (x, y) in zip(chain[s1:e1], uni[s2:e2]) if x == y))
    return ident / len(chain)
extract_alkaline_structures_OUT = extract_alkaline_structures_HERE / 'outputs'
CIF_CACHE = extract_alkaline_structures_OUT / '_cif_cache'
extract_alkaline_structures_STRUCT = extract_alkaline_structures_OUT / 'structures'
AF_CACHE = extract_alkaline_structures_OUT / '_af_cache'
extract_alkaline_structures_MANIFEST = extract_alkaline_structures_OUT / 'alkaline_structures_manifest.csv'
extract_alkaline_structures_PARTIAL = extract_alkaline_structures_OUT / 'alkaline_structures_manifest.partial.csv'
IDENTITY_MIN = 0.9
(LEN_MIN, LEN_MAX) = (40, 700)
MAX_CIF_MB = 60
AF_URL = 'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb'
BB = ['N', 'CA', 'C', 'O']
def _write_chain(sel, dest):
    sel = sel.copy()
    sel.chain_id[:] = 'A'
    pf = pdbio.PDBFile()
    pf.set_structure(sel)
    pf.write(str(dest))
def extract_pdb(acc, pdb_id, chain, uni, dest):
    pdb_id = str(pdb_id).strip().upper()
    chain = str(chain).strip()
    cif = CIF_CACHE / f'{pdb_id}.cif'
    if not cif.exists():
        rcsb.fetch(pdb_id, 'cif', str(CIF_CACHE))
    if cif.stat().st_size > MAX_CIF_MB * 1000000.0:
        return (None, None, 'large_assembly_skipped')
    arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    sel = aa[aa.chain_id == chain]
    seq = ''.join((three_to_one(r) for r in sel[sel.atom_name == 'CA'].res_name)) if sel.array_length() else ''
    idn = gapped_identity(seq, uni)
    if idn < IDENTITY_MIN:
        best = (None, 0.0, '')
        for ch in sorted(set(aa.chain_id)):
            ca = aa[(aa.chain_id == ch) & (aa.atom_name == 'CA')]
            if ca.array_length() == 0:
                continue
            s = ''.join((three_to_one(r) for r in ca.res_name))
            i = gapped_identity(s, uni)
            if i > best[1]:
                best = (ch, i, s)
        (chain, idn, seq) = best
        if chain is None:
            return (None, None, 'no_protein_chain')
        sel = aa[aa.chain_id == chain]
    if idn < IDENTITY_MIN:
        return (None, round(idn, 4), 'identity_low')
    _write_chain(sel, dest)
    return (seq, round(idn, 4), None)
def af_pdb_url(acc):
    api = json.load(urllib.request.urlopen(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}', timeout=60))
    return api[0]['pdbUrl']
def extract_af(acc, uni, dest):
    cached = AF_CACHE / f'AF-{acc}.pdb'
    if not cached.exists():
        urllib.request.urlretrieve(af_pdb_url(acc), str(cached))
    arr = pdbio.get_structure(pdbio.PDBFile.read(str(cached)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    ca = aa[aa.atom_name == 'CA']
    seq = ''.join((three_to_one(r) for r in ca.res_name))
    idn = gapped_identity(seq, uni)
    if idn < IDENTITY_MIN:
        return (None, round(idn, 4), 'identity_low')
    _write_chain(aa, dest)
    return (seq, round(idn, 4), None)
def extract_alkaline_structures_process(rec):
    (acc, role) = (rec['acc'], rec['role'])
    dest = extract_alkaline_structures_STRUCT / ('cases' if role == 'case' else 'controls') / f'{acc}.pdb'
    out = dict(rec)
    out['chain_pdb_path'] = ''
    out['extracted_seq'] = ''
    out['extracted_identity'] = np.nan
    out['fail_reason'] = None
    try:
        if dest.exists() and dest.stat().st_size > 0:
            out['chain_pdb_path'] = str(dest)
            out['fail_reason'] = None
            return out
        if rec['structure_source'] == 'PDB':
            (seq, idn, fail) = extract_pdb(acc, rec['pdb_id'], rec['pdb_chain'], rec['sequence'], dest)
        else:
            (seq, idn, fail) = extract_af(acc, rec['sequence'], dest)
        out['extracted_identity'] = idn
        out['fail_reason'] = fail
        if not fail:
            out['chain_pdb_path'] = str(dest)
            out['extracted_seq'] = seq
    except Exception as ex:
        out['fail_reason'] = f'error:{str(ex)[:50]}'
    return out
def backbone_record(path, name, extra):
    arr = pdbio.get_structure(pdbio.PDBFile.read(str(path)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    aa = aa[aa.chain_id == 'A']
    res_ids = []
    for rid in aa.res_id:
        if not res_ids or res_ids[-1] != rid:
            res_ids.append(rid)
    coords = {b: [] for b in BB}
    seq = []
    for rid in res_ids:
        r = aa[aa.res_id == rid]
        seq.append(three_to_one(r.res_name[0]))
        for b in BB:
            at = r[r.atom_name == b]
            coords[b].append([float(x) for x in at.coord[0]] if at.array_length() else [float('nan')] * 3)
    s = ''.join(seq)
    rec = {'name': name, 'num_of_chains': 1, 'seq': s, 'seq_chain_A': s, 'coords_chain_A': {f'{b}_chain_A': coords[b] for b in BB}}
    rec.update(extra)
    return rec
def extract_alkaline_structures_collect(tag):
    ca = pd.read_csv(extract_alkaline_structures_OUT / f'alkaline_optimum_cases_{tag}_stageD.csv')
    co = pd.read_csv(extract_alkaline_structures_OUT / f'matched_neutral_controls_for_{tag}_cases_stageD.csv')
    rows = []
    for (df, role) in [(ca, 'case'), (co, 'control')]:
        d = df[df.qc_pass & df.split.notna()].copy()
        for (_, r) in d.iterrows():
            rows.append({'acc': r.acc, 'role': role, 'set': tag, 'split': r.split, 'structure_source': r.structure_source, 'pdb_id': r.get('pdb_id', ''), 'pdb_chain': r.get('pdb_chain', ''), 'cluster_id': r.get('cluster_id', ''), 'sequence': r.sequence, 'label': 1 if role == 'case' else 0})
    return rows
def extract_alkaline_structures_main():
    tags = sys.argv[1:] or ['high_confidence']
    rows = [r for t in tags for r in extract_alkaline_structures_collect(t)]
    seen = {}
    for r in rows:
        seen[r['acc'], r['role']] = r
    rows = list(seen.values())
    print(f"structures to extract: {len(rows)}  (PDB {sum((r['structure_source'] == 'PDB' for r in rows))}, AF {sum((r['structure_source'] == 'AF' for r in rows))})")
    done = {}
    if extract_alkaline_structures_PARTIAL.exists():
        for r in pd.read_csv(extract_alkaline_structures_PARTIAL).to_dict('records'):
            done[r['acc'], r['role']] = r
    todo = [r for r in rows if (r['acc'], r['role']) not in done]
    results = list(done.values())
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(extract_alkaline_structures_process, r) for r in todo]
        for (i, f) in enumerate(tqdm(as_completed(futs), total=len(futs), desc='extract')):
            results.append(f.result())
            if i % 40 == 0:
                pd.DataFrame(results).to_csv(extract_alkaline_structures_PARTIAL, index=False)
    man = pd.DataFrame(results)
    man.to_csv(extract_alkaline_structures_MANIFEST, index=False)
    if extract_alkaline_structures_PARTIAL.exists():
        extract_alkaline_structures_PARTIAL.unlink()
    pth = man.chain_pdb_path.astype(str)
    ok = man[man.fail_reason.isna() & (pth.str.len() > 0) & (pth != 'nan')].copy()
    print(f'\nextracted OK: {len(ok)}/{len(man)}')
    print('by source:', dict(ok.structure_source.value_counts()))
    print('by role  :', dict(ok.role.value_counts()))
    if man.fail_reason.notna().any():
        print('failures :', dict(man.fail_reason.value_counts()))
    for tag in tags:
        ca = pd.read_csv(extract_alkaline_structures_OUT / f'alkaline_optimum_cases_{tag}_stageD.csv')
        co = pd.read_csv(extract_alkaline_structures_OUT / f'matched_neutral_controls_for_{tag}_cases_stageD.csv')
        L = {**dict(zip(ca.acc, ca.length)), **dict(zip(co.acc, co.length))}
        okset = ok[ok.set == tag].set_index(['acc', 'role'])
        pairs = ca[ca.qc_pass & ca.split.notna()]
        n = {'train': 0, 'val': 0, 'test': 0}
        skipped = 0
        skip_len = 0
        writers = {sp: open(extract_alkaline_structures_OUT / f'alkaline_parsed_{tag}_{sp}.jsonl', 'w') for sp in n}
        for (_, p) in pairs.iterrows():
            (kc, kk) = ((p.acc, 'case'), (p.matched_control_uniprot, 'control'))
            if kc not in okset.index or kk not in okset.index:
                skipped += 1
                continue
            (lc, lk) = (L.get(p.acc), L.get(p.matched_control_uniprot))
            if not (lc and lk and (LEN_MIN <= lc <= LEN_MAX) and (LEN_MIN <= lk <= LEN_MAX)):
                skip_len += 1
                continue
            sp = p.split
            for (acc, role, key) in [(p.acc, 'case', kc), (p.matched_control_uniprot, 'control', kk)]:
                extra = {'role': role, 'label': 1 if role == 'case' else 0, 'split': sp, 'set': tag, 'cluster_id': p.cluster_id, 'pair_case': p.acc}
                try:
                    rec = backbone_record(okset.loc[key].chain_pdb_path, acc, extra)
                    writers[sp].write(json.dumps(rec) + '\n')
                    n[sp] += 1
                except Exception as ex:
                    print(f'   jsonl skip {acc}: {str(ex)[:40]}')
        for w in writers.values():
            w.close()
        print(f"[{tag}] jsonl chains written: {n} ({n['train'] // 2}+{n['val'] // 2}+{n['test'] // 2} pairs); skipped: missing structure {skipped}, length 40-700 {skip_len}")
def extract_alkaline_structures__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(extract_alkaline_structures_HERE))
    CIF_CACHE.mkdir(parents=True, exist_ok=True)
    (extract_alkaline_structures_STRUCT / 'cases').mkdir(parents=True, exist_ok=True)
    (extract_alkaline_structures_STRUCT / 'controls').mkdir(parents=True, exist_ok=True)
    AF_CACHE.mkdir(parents=True, exist_ok=True)
    extract_alkaline_structures_main()

# ---------- from extract_alkaliphile_structures.py ----------
extract_alkaliphile_structures_HERE = Path(__file__).resolve().parent
extract_alkaliphile_structures_OUT = extract_alkaliphile_structures_HERE / 'outputs'
extract_alkaliphile_structures_STRUCT = extract_alkaliphile_structures_MANIFEST = extract_alkaliphile_structures_PARTIAL = None
def extract_alkaliphile_structures_process(rec):
    (acc, role) = (rec['acc'], rec['role'])
    dest = extract_alkaliphile_structures_STRUCT / ('cases' if role == 'case' else 'controls') / f'{acc}.pdb'
    out = dict(rec)
    out['chain_pdb_path'] = ''
    out['extracted_seq'] = ''
    out['extracted_identity'] = np.nan
    out['fail_reason'] = None
    try:
        if dest.exists() and dest.stat().st_size > 0:
            out['chain_pdb_path'] = str(dest)
            return out
        if rec['structure_source'] == 'PDB':
            (seq, idn, fail) = extract_pdb(acc, rec['pdb_id'], rec['pdb_chain'], rec['sequence'], dest)
        elif rec['structure_source'] == 'AF':
            (seq, idn, fail) = extract_af(acc, rec['sequence'], dest)
        else:
            (seq, idn, fail) = (None, np.nan, 'no_structure_source')
        out['extracted_identity'] = idn
        out['fail_reason'] = fail
        if not fail:
            out['chain_pdb_path'] = str(dest)
            out['extracted_seq'] = seq
    except Exception as ex:
        out['fail_reason'] = f'error:{str(ex)[:50]}'
    return out
def extract_alkaliphile_structures_collect(C):
    ca = pd.read_csv(extract_alkaliphile_structures_OUT / C['cases_D'])
    co = pd.read_csv(extract_alkaliphile_structures_OUT / C['ctrls_D'])
    rows = []
    for (df, role) in [(ca, 'case'), (co, 'control')]:
        d = df[(df.qc_pass == True) & df.split.notna()].copy()
        for (_, r) in d.iterrows():
            rows.append({'acc': r.acc, 'role': role, 'split': r.split, 'structure_source': r.structure_source, 'pdb_id': r.get('pdb_id', ''), 'pdb_chain': r.get('pdb_chain', ''), 'cluster_id': r.get('cluster_id', ''), 'group': r.group, 'clade': r.clade, 'sequence': r.sequence, 'tight_pair': r.get('tight_pair', r.get('bacillaceae_pair', False)), 'label': 1 if role == 'case' else 0})
    return (rows, ca, co)
def extract_alkaliphile_structures_main():
    global STRUCT, MANIFEST, PARTIAL
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='alkaline', choices=['alkaline', 'acid'])
    a = ap.parse_args()
    C = cfg(a.cohort)
    extract_alkaliphile_structures_STRUCT = extract_alkaliphile_structures_OUT / C['struct_dir']
    (extract_alkaliphile_structures_STRUCT / 'cases').mkdir(parents=True, exist_ok=True)
    (extract_alkaliphile_structures_STRUCT / 'controls').mkdir(parents=True, exist_ok=True)
    extract_alkaliphile_structures_MANIFEST = extract_alkaliphile_structures_OUT / C['manifest']
    extract_alkaliphile_structures_PARTIAL = extract_alkaliphile_structures_OUT / C['manifest'].replace('.csv', '.partial.csv')
    (rows, ca, co) = extract_alkaliphile_structures_collect(C)
    seen = {}
    for r in rows:
        seen[r['acc'], r['role']] = r
    rows = list(seen.values())
    print(f"structures to extract: {len(rows)}  (PDB {sum((r['structure_source'] == 'PDB' for r in rows))}, AF {sum((r['structure_source'] == 'AF' for r in rows))})")
    done = {}
    if extract_alkaliphile_structures_PARTIAL.exists():
        for r in pd.read_csv(extract_alkaliphile_structures_PARTIAL).to_dict('records'):
            done[r['acc'], r['role']] = r
    todo = [r for r in rows if (r['acc'], r['role']) not in done]
    results = list(done.values())
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(extract_alkaliphile_structures_process, r) for r in todo]
        for (i, f) in enumerate(tqdm(as_completed(futs), total=len(futs), desc='extract')):
            results.append(f.result())
            if i % 40 == 0:
                pd.DataFrame(results).to_csv(extract_alkaliphile_structures_PARTIAL, index=False)
    man = pd.DataFrame(results)
    man.to_csv(extract_alkaliphile_structures_MANIFEST, index=False)
    if extract_alkaliphile_structures_PARTIAL.exists():
        extract_alkaliphile_structures_PARTIAL.unlink()
    pth = man.chain_pdb_path.astype(str)
    ok = man[man.fail_reason.isna() & (pth.str.len() > 0) & (pth != 'nan')].copy()
    print(f'\nextracted OK: {len(ok)}/{len(man)} | by source {dict(ok.structure_source.value_counts())} | by role {dict(ok.role.value_counts())}')
    if man.fail_reason.notna().any():
        print('failures:', dict(man.fail_reason.value_counts()))
    L = {**dict(zip(ca.acc, ca.length)), **dict(zip(co.acc, co.length))}
    okset = ok.set_index(['acc', 'role'])
    pairs = ca[(ca.qc_pass == True) & ca.split.notna()]
    n = {'train': 0, 'val': 0, 'test': 0}
    skip_struct = skip_len = 0
    writers = {sp: open(extract_alkaliphile_structures_OUT / f"{C['jsonl_prefix']}_parsed_{sp}.jsonl", 'w') for sp in n}
    for (_, p) in pairs.iterrows():
        (kc, kk) = ((p.acc, 'case'), (p.matched_control_uniprot, 'control'))
        if kc not in okset.index or kk not in okset.index:
            skip_struct += 1
            continue
        (lc, lk) = (L.get(p.acc), L.get(p.matched_control_uniprot))
        if not (lc and lk and (LEN_MIN <= lc <= LEN_MAX) and (LEN_MIN <= lk <= LEN_MAX)):
            skip_len += 1
            continue
        sp = p.split
        for (acc, role, key) in [(p.acc, 'case', kc), (p.matched_control_uniprot, 'control', kk)]:
            row = okset.loc[key]
            extra = {'role': role, 'label': 1 if role == 'case' else 0, 'split': sp, 'cluster_id': p.cluster_id, 'pair_case': p.acc, 'group': row.group, 'clade': row.clade, 'tight_pair': bool(p.get('tight_pair', False))}
            try:
                writers[sp].write(json.dumps(backbone_record(row.chain_pdb_path, acc, extra)) + '\n')
                n[sp] += 1
            except Exception as ex:
                print(f'   jsonl skip {acc}: {str(ex)[:40]}')
    for w in writers.values():
        w.close()
    print(f"jsonl chains: {n} ({n['train'] // 2}+{n['val'] // 2}+{n['test'] // 2} pairs); skipped: no_struct {skip_struct}, length {skip_len}")
def extract_alkaliphile_structures__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(extract_alkaliphile_structures_HERE))
    extract_alkaliphile_structures_main()

# ---------- from stage_c_chain_qc.py ----------
stage_c_chain_qc_HERE = Path(__file__).resolve().parent
stage_c_chain_qc_OUT = stage_c_chain_qc_HERE / 'outputs'
FILES = ['alkaline_optimum_cases_high_confidence', 'matched_neutral_controls_for_high_confidence_cases', 'alkaline_optimum_cases_inclusive', 'matched_neutral_controls_for_inclusive_cases']
(stage_c_chain_qc_COV_PASS, stage_c_chain_qc_COV_BORDER) = (0.8, 0.7)
(stage_c_chain_qc_RES_XRAY, stage_c_chain_qc_RES_EM) = (3.0, 4.0)
(LEN_MIN, LEN_MAX) = (40, 700)
def parse_xref(props):
    d = {p['key']: p['value'] for p in props}
    method = d.get('Method', '')
    res = None
    mr = re.search('([\\d.]+)\\s*A', d.get('Resolution', '') or '')
    if mr:
        res = float(mr.group(1))
    spans = re.findall('=(\\d+)-(\\d+)', d.get('Chains', '') or '')
    chain = d.get('Chains', '').split('=')[0].split('/')[0] if '=' in d.get('Chains', '') else ''
    if spans:
        st = min((int(a) for (a, _) in spans))
        en = max((int(b) for (_, b) in spans))
    else:
        st = en = None
    return (method, res, chain, st, en)
def best_chain(xrefs, ulen):
    best = None
    for x in xrefs:
        if x['database'] != 'PDB':
            continue
        (method, res, chain, st, en) = parse_xref(x.get('properties', []))
        if st is None or not ulen:
            continue
        cov = (en - st + 1) / ulen
        rank_method = {'X-ray': 0, 'EM': 1, 'NMR': 2}.get(method, 3)
        score = (rank_method, -cov, res if res is not None else 9.9)
        rec = dict(pdb_id=x['id'], chain=chain, method=method, res=res, start=st, end=en, cov=round(cov, 3))
        if best is None or score < best[0]:
            best = (score, rec)
    return best[1] if best else None
def qc(rec):
    if rec is None:
        return (False, 'no_mapped_chain')
    flags = []
    if rec['method'] == 'X-ray':
        if rec['res'] is None or rec['res'] > stage_c_chain_qc_RES_XRAY:
            flags.append(f'res>{stage_c_chain_qc_RES_XRAY}')
    elif rec['method'] == 'EM':
        if rec['res'] is None or rec['res'] > stage_c_chain_qc_RES_EM:
            flags.append(f'EM_res>{stage_c_chain_qc_RES_EM}')
    elif rec['method'] != 'NMR':
        flags.append(f"method={rec['method'] or '?'}")
    if rec['cov'] < stage_c_chain_qc_COV_BORDER:
        flags.append('coverage<0.70')
    elif rec['cov'] < stage_c_chain_qc_COV_PASS:
        flags.append('coverage_borderline')
    span = rec['end'] - rec['start'] + 1
    if not LEN_MIN <= span <= LEN_MAX:
        flags.append('length_out_of_range')
    hard = [f for f in flags if f != 'coverage_borderline']
    return (len(hard) == 0, ';'.join(flags))
def fetch_xrefs(accs):
    import time
    out = {}
    url = 'https://rest.uniprot.org/uniprotkb/search'
    for i in range(0, len(accs), 50):
        chunk = accs[i:i + 50]
        params = {'query': 'accession:(' + ' OR '.join(chunk) + ')', 'format': 'json', 'size': 50, 'fields': 'accession,length,xref_pdb'}
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=60)
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        for e in r.json().get('results', []):
            out[e['primaryAccession']] = (e.get('uniProtKBCrossReferences', []), e.get('sequence', {}).get('length'))
        print(f'  fetched {min(i + 50, len(accs))}/{len(accs)}', flush=True)
    return out
def stage_c_chain_qc_main():
    import pandas as pd
    dfs = {f: pd.read_csv(stage_c_chain_qc_OUT / f'{f}.csv') for f in FILES}
    accs = sorted({a for df in dfs.values() for a in df.acc})
    print(f'fetching PDB xrefs for {len(accs)} unique accessions ...')
    xr = fetch_xrefs(accs)
    chainmap = {}
    for (acc, (xrefs, ulen)) in xr.items():
        rec = best_chain(xrefs, ulen)
        (ok, flags) = qc(rec)
        chainmap[acc] = (rec, ok, flags, ulen)
    for (f, df) in dfs.items():
        rows = []
        for (_, r) in df.iterrows():
            (rec, ok, flags, ulen) = chainmap.get(r.acc, (None, False, 'not_found', None))
            d = r.to_dict()
            if rec:
                d.update(pdb_id=rec['pdb_id'], pdb_chain=rec['chain'], method=rec['method'], resolution=rec['res'], uniprot_start=rec['start'], uniprot_end=rec['end'], chain_coverage_fraction=rec['cov'])
            d['qc_pass'] = ok
            d['qc_flags'] = flags
            if 'label_confidence' in d and rec and ('coverage_borderline' in flags) and (d['label_confidence'] == 'high'):
                d['label_confidence'] = 'medium'
            rows.append(d)
        out = pd.DataFrame(rows)
        out.to_csv(stage_c_chain_qc_OUT / f'{f}_stageC.csv', index=False)
        npass = out.qc_pass.sum()
        print(f'  {f}: {npass}/{len(out)} pass QC')
        if 'matched_case_uniprot' not in out.columns and len(out):
            pass
    for tag in ['high_confidence', 'inclusive']:
        ca = pd.read_csv(stage_c_chain_qc_OUT / f'alkaline_optimum_cases_{tag}_stageC.csv')
        co = pd.read_csv(stage_c_chain_qc_OUT / f'matched_neutral_controls_for_{tag}_cases_stageC.csv')
        cok = set(ca[ca.qc_pass].acc)
        ctrl_ok = set(co[co.qc_pass].acc)
        pairs = ca[ca.qc_pass & ca.matched_control_uniprot.isin(ctrl_ok) & (ca.matched_control_uniprot != '')]
        print(f'[{tag}] cases pass {len(cok)} | controls pass {len(ctrl_ok)} | BOTH-pass matched pairs {len(pairs)}')
        if tag == 'high_confidence':
            import collections
            print('   case qc_flags:', dict(collections.Counter(';'.join(ca[~ca.qc_pass].qc_flags.fillna('')).split(';')).most_common(6)))
def stage_c_chain_qc__entry():
    stage_c_chain_qc_main()

# ---------- from stage_d_alkaliphile.py ----------
stage_d_alkaliphile_HERE = Path(__file__).resolve().parent
stage_d_alkaliphile_OUT = stage_d_alkaliphile_HERE / 'outputs'
def stage_d_alkaliphile_cluster_and_split(C, target=(0.7, 0.15, 0.15)):
    ca = pd.read_csv(stage_d_alkaliphile_OUT / C['cases_C'])
    co = pd.read_csv(stage_d_alkaliphile_OUT / C['ctrls_C'])
    ca['matched_control_uniprot'] = ca.matched_control_uniprot.fillna('').astype(str)
    pairs = ca[ca.matched_control_uniprot.str.len() > 0].copy()
    cseq = dict(zip(ca.acc, ca.sequence))
    oseq = dict(zip(co.acc, co.sequence))
    cbac = dict(zip(ca.acc, ca[C['tight_in']]))
    obac = dict(zip(co.acc, co[C['tight_in']]))
    seqs = {a: cseq[a] for a in pairs.acc} | {r.matched_control_uniprot: oseq[r.matched_control_uniprot] for r in pairs.itertuples() if r.matched_control_uniprot in oseq}
    accs = list(seqs)
    km = {a: kmers(seqs[a]) for a in accs}
    print(f'usable pairs {len(pairs)} | sequences {len(accs)}')
    uf = UF(accs)
    n_align = 0
    for i in range(len(accs)):
        ai = accs[i]
        ki = km[ai]
        for j in range(i + 1, len(accs)):
            aj = accs[j]
            if len(ki & km[aj]) / max(1, min(len(ki), len(km[aj]))) < 0.1:
                continue
            n_align += 1
            if pid(seqs[ai], seqs[aj]) >= ID_THRESH:
                uf.union(ai, aj)
    for r in pairs.itertuples():
        if r.matched_control_uniprot in seqs:
            uf.union(r.acc, r.matched_control_uniprot)
    clusters = collections.defaultdict(list)
    for a in accs:
        clusters[uf.find(a)].append(a)
    cl_ids = {a: f"{C['cl_prefix']}_cl{idx}" for (idx, members) in enumerate(sorted(clusters.values(), key=len, reverse=True)) for a in members}
    print(f'{len(clusters)} clusters at {int(ID_THRESH * 100)}% identity ({n_align} alignments); largest cluster {max((len(v) for v in clusters.values()))} seqs')
    groups = collections.defaultdict(list)
    for r in pairs.itertuples():
        groups[uf.find(r.acc)].append(r.acc)
    npair = len(pairs)
    tgt = {s: t * npair for (s, t) in zip(['train', 'val', 'test'], target)}
    cur = {'train': 0, 'val': 0, 'test': 0}
    assign = {}
    for g in sorted(groups, key=lambda g: -len(groups[g])):
        s = min(['train', 'val', 'test'], key=lambda s: cur[s] - tgt[s])
        assign[g] = s
        cur[s] += len(groups[g])
    pair_split = {a: assign[uf.find(a)] for a in pairs.acc}
    ident = {r.acc: round(pid(cseq[r.acc], oseq[r.matched_control_uniprot]), 3) for r in pairs.itertuples() if r.matched_control_uniprot in oseq}
    bpair = {r.acc: bool(cbac.get(r.acc)) and bool(obac.get(r.matched_control_uniprot)) for r in pairs.itertuples()}
    ca['cluster_id'] = ca.acc.map(cl_ids)
    ca['split'] = ca.acc.map(pair_split)
    ca['sequence_identity_to_control'] = ca.acc.map(ident)
    ca['tight_pair'] = ca.acc.map(bpair)
    co['cluster_id'] = co.acc.map(cl_ids)
    co['split'] = co.matched_case_uniprot.map(pair_split)
    co['tight_pair'] = co.matched_case_uniprot.map(bpair)
    ca.to_csv(stage_d_alkaliphile_OUT / C['cases_D'], index=False)
    co.to_csv(stage_d_alkaliphile_OUT / C['ctrls_D'], index=False)
    sp = ca.dropna(subset=['split'])
    print('split (pairs):', dict(sp.split.value_counts()))
    print(f'case-control identity: median {sp.sequence_identity_to_control.median():.2f} (>40% pairs: {(sp.sequence_identity_to_control > 0.4).sum()})')
    print(f'tight-family-only pairs (sensitivity subset): {int(sp.tight_pair.sum())}')
    print('  by split:', dict(sp[sp.tight_pair].split.value_counts()))
def stage_d_alkaliphile__entry():
    sys.path.insert(0, str(stage_d_alkaliphile_HERE))
    random.seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='alkaline', choices=['alkaline', 'acid'])
    a = ap.parse_args()
    stage_d_alkaliphile_cluster_and_split(cfg(a.cohort))
    print(f'\nWrote {a.cohort} *_stageD.csv. Train on matched cases (Option 3), hold out val/test clusters; tight_pair subset = phylogeny-controlled sensitivity.')

# ---------- from stage_d_cluster_split.py ----------
stage_d_cluster_split_HERE = Path(__file__).resolve().parent
stage_d_cluster_split_OUT = stage_d_cluster_split_HERE / 'outputs'
ID_THRESH = 0.4
aligner = Align.PairwiseAligner(mode='global', match_score=1, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5)
def kmers(s, k=5):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}
def pid(a, b):
    if not a or not b:
        return 0.0
    aln = aligner.align(a, b)[0]
    ident = 0
    for ((s1, e1), (s2, e2)) in zip(*aln.aligned):
        ident += sum((1 for (x, y) in zip(a[s1:e1], b[s2:e2]) if x == y))
    return ident / min(len(a), len(b))
class UF:

    def __init__(s, items):
        s.p = {i: i for i in items}

    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]
            x = s.p[x]
        return x

    def union(s, a, b):
        s.p[s.find(a)] = s.find(b)
def stage_d_cluster_split_cluster_and_split(tag, target=(0.7, 0.15, 0.15)):
    ca = pd.read_csv(stage_d_cluster_split_OUT / f'alkaline_optimum_cases_{tag}_stageC.csv')
    co = pd.read_csv(stage_d_cluster_split_OUT / f'matched_neutral_controls_for_{tag}_cases_stageC.csv')
    ctrl_ok = set(co[co.qc_pass].acc)
    pairs = ca[ca.qc_pass & ca.matched_control_uniprot.isin(ctrl_ok) & (ca.matched_control_uniprot != '')].copy()
    cseq = dict(zip(ca.acc, ca.sequence))
    oseq = dict(zip(co.acc, co.sequence))
    seqs = {a: cseq[a] for a in pairs.acc} | {r.matched_control_uniprot: oseq[r.matched_control_uniprot] for r in pairs.itertuples()}
    accs = list(seqs)
    km = {a: kmers(seqs[a]) for a in accs}
    print(f'[{tag}] usable pairs {len(pairs)} | sequences {len(accs)}')
    uf = UF(accs)
    n_align = 0
    for i in range(len(accs)):
        ai = accs[i]
        ki = km[ai]
        for j in range(i + 1, len(accs)):
            aj = accs[j]
            inter = len(ki & km[aj])
            if inter / max(1, min(len(ki), len(km[aj]))) < 0.1:
                continue
            n_align += 1
            if pid(seqs[ai], seqs[aj]) >= ID_THRESH:
                uf.union(ai, aj)
    for r in pairs.itertuples():
        uf.union(r.acc, r.matched_control_uniprot)
    clusters = collections.defaultdict(list)
    for a in accs:
        clusters[uf.find(a)].append(a)
    cl_ids = {a: f'{tag[:2]}_cl{idx}' for (idx, members) in enumerate(sorted(clusters.values(), key=len, reverse=True)) for a in members}
    print(f'   {len(clusters)} clusters at {int(ID_THRESH * 100)}% identity ({n_align} alignments); largest cluster {max((len(v) for v in clusters.values()))} seqs')
    pair_cluster = {r.acc: uf.find(r.acc) for r in pairs.itertuples()}
    groups = collections.defaultdict(list)
    for r in pairs.itertuples():
        groups[uf.find(r.acc)].append(r.acc)
    gids = list(groups)
    random.shuffle(gids)
    npair = len(pairs)
    tgt = {s: t * npair for (s, t) in zip(['train', 'val', 'test'], target)}
    assign = {}
    cur = {'train': 0, 'val': 0, 'test': 0}
    for g in sorted(gids, key=lambda g: -len(groups[g])):
        s = min(['train', 'val', 'test'], key=lambda s: cur[s] - tgt[s])
        assign[g] = s
        cur[s] += len(groups[g])
    pair_split = {ca_acc: assign[uf.find(ca_acc)] for ca_acc in pairs.acc}
    v12 = set(pd.read_csv(stage_d_cluster_split_HERE.parent / 'dataset_update' / 'main_plus_r2_r3_analysis_v12_corrected.csv', usecols=['Entry'], low_memory=False).Entry)
    ident = {r.acc: round(pid(cseq[r.acc], oseq[r.matched_control_uniprot]), 3) for r in pairs.itertuples()}
    ca['cluster_id'] = ca.acc.map(cl_ids)
    ca['split'] = ca.acc.map(pair_split)
    ca['sequence_identity_to_control'] = ca.acc.map(ident)
    ca['in_v12'] = ca.acc.isin(v12)
    co['cluster_id'] = co.acc.map(cl_ids)
    co['split'] = co.matched_case_uniprot.map(pair_split)
    ca.to_csv(stage_d_cluster_split_OUT / f'alkaline_optimum_cases_{tag}_stageD.csv', index=False)
    co.to_csv(stage_d_cluster_split_OUT / f'matched_neutral_controls_for_{tag}_cases_stageD.csv', index=False)
    sp = ca.dropna(subset=['split'])
    print('   split (pairs):', dict(sp.split.value_counts()))
    print(f'   case-control identity: median {sp.sequence_identity_to_control.median():.2f} (>40% pairs: {(sp.sequence_identity_to_control > 0.4).sum()})')
    print(f'   in_v12 cases (exclude from training or from external eval): {int(sp.in_v12.sum())}')
    return pairs
def stage_d_cluster_split_main():
    for tag in ['high_confidence', 'inclusive']:
        stage_d_cluster_split_cluster_and_split(tag)
    print('\nWrote *_stageD.csv. Recommendation: train on the matched cases (Option 3),')
    print('hold out val/test clusters, use v12 (minus in_v12 cases) as the external eval.')
def stage_d_cluster_split__entry():
    random.seed(0)
    stage_d_cluster_split_main()

# ---------- from prep_secreted_targets.py ----------
prep_secreted_targets_HERE = Path(__file__).resolve().parent
ROOT = prep_secreted_targets_HERE.parents[1]
TEST_JSONL = ROOT / 'finetune' / 'data' / 'alkaliphile_parsed_test.jsonl'
STRUCT_DIR = ROOT / 'finetune' / 'data' / 'structures_alkaliphile' / 'controls'
prep_secreted_targets_OUT = ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'generation'
N_TARGETS = 10
(LEN_MIN, LEN_MAX) = (50, 300)
MAX_SURF_FRAC = 0.85
def prep_secreted_targets_main():
    prep_secreted_targets_OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(TEST_JSONL)]
    ctrl = sorted([r for r in rows if r['role'] == 'control'], key=lambda r: r['name'])
    (targets, surf_map) = ([], {})
    for r in ctrl:
        if len(targets) >= N_TARGETS:
            break
        (name, seq) = (r['name'], r['seq'])
        if not LEN_MIN <= len(seq) <= LEN_MAX:
            continue
        pdb = STRUCT_DIR / f'{name}.pdb'
        if not pdb.exists():
            continue
        try:
            (letters, rsa) = per_residue_rsa(pdb)
        except Exception as e:
            print(f'  skip {name}: RSA failed ({e})')
            continue
        if len(rsa) != len(seq):
            print(f'  skip {name}: structure {len(rsa)} != seq {len(seq)}')
            continue
        surf = [i for (i, v) in enumerate(rsa) if not np.isnan(v) and v >= RSA_CUT]
        if len(surf) / len(seq) > MAX_SURF_FRAC:
            print(f'  skip {name}: {len(surf)}/{len(seq)} surface (disordered/extended)')
            continue
        targets.append(dict(name=name, seq=seq, structure_path=str(pdb), n_res=len(seq), n_surface=len(surf)))
        surf_map[name] = {'surface': surf, 'len': len(seq)}
    if len(targets) < N_TARGETS:
        print(f'WARNING: only {len(targets)} targets passed QC (wanted {N_TARGETS})')
    pd.DataFrame(targets).to_csv(prep_secreted_targets_OUT / 'secreted_targets.csv', index=False)
    json.dump(surf_map, open(prep_secreted_targets_OUT / 'secreted_surface_positions.json', 'w'))
    print(f'wrote {len(targets)} targets -> secreted_targets.csv + secreted_surface_positions.json')
    print(pd.DataFrame(targets)[['name', 'n_res', 'n_surface']].to_string(index=False))
def prep_secreted_targets__entry():
    sys.path.insert(0, str(ROOT / 'design'))
    prep_secreted_targets_main()

_STEPS = {
    'build-dataset': build_dataset__entry,
    'build-alkaline-dataset': build_alkaline_dataset__entry,
    'assign-structures-alkaliphile': assign_structures_alkaliphile__entry,
    'extract-alkaline-structures': extract_alkaline_structures__entry,
    'extract-alkaliphile-structures': extract_alkaliphile_structures__entry,
    'stage-c-chain-qc': stage_c_chain_qc__entry,
    'stage-d-alkaliphile': stage_d_alkaliphile__entry,
    'stage-d-cluster-split': stage_d_cluster_split__entry,
    'prep-secreted-targets': prep_secreted_targets__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

