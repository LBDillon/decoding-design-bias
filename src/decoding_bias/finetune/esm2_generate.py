"""decoding_bias.finetune.esm2_generate -- merged provenance (see ARCHIVE_MAP.md).

Sections:
  - run_generation_local
  - score_esm2_masked_marginals
  - analyse_esm2_score_shifts
  - esm_design_surface
  - esm_design_heatmaps
"""

from __future__ import annotations

import argparse
import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
from pathlib import Path
from esm_generation import esm_position_distributions, make_esm_predict_fn, iterative_infill, CANONICAL_AA, acidic_propensity, ACIDIC, BASIC
from typing import Iterable, List, Optional, Dict, Tuple
from surface_features_alkaline import per_residue_rsa, comp, MAXASA, RSA_CUT

# ---------- from run_generation_local.py ----------
run_generation_local_HERE = Path(__file__).resolve().parent
run_generation_local_ROOT = run_generation_local_HERE.parents[1]
BASE_MODEL = 'facebook/esm2_t12_35M_UR50D'
DATA = run_generation_local_ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'data'
run_generation_local_GEN_DIR = run_generation_local_ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'generation'
RUNS = run_generation_local_HERE / 'runs_local'
TEMPLATE_IDS = ['Q4R312', 'P0A7R6', 'A9A498', 'A0A1D8PCG7', 'Q9Z9J6']
ARMS = {'base': None, 'AlkSecESM35M': 'alkaline_case', 'NeuSecESM35M': 'alkaline_neu'}
def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
def load_seqs(stem, split):
    return pd.read_csv(DATA / f'{stem}_{split}.csv')['sequence'].dropna().tolist()
def finetune_mlm(train_seqs, out_dir, epochs, lr, device, mlm_prob=0.15, batch=4, max_len=1022, seed=1):
    """Minimal continued-MLM fine-tune (BERT 80/10/10 masking), no HF Trainer."""
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    vocab = tok.vocab_size
    model.train()
    for ep in range(epochs):
        seqs = train_seqs[:]
        rng.shuffle(seqs)
        (tot, n) = (0.0, 0)
        for i in range(0, len(seqs), batch):
            enc = tok(seqs[i:i + batch], return_tensors='pt', padding=True, truncation=True, max_length=max_len, return_special_tokens_mask=True)
            ids = enc['input_ids'].to(device)
            attn = enc['attention_mask'].to(device)
            special = enc['special_tokens_mask'].bool().to(device)
            labels = ids.clone()
            probs = torch.full(ids.shape, mlm_prob, device=device)
            probs[special] = 0.0
            masked = torch.bernoulli(probs).bool()
            labels[~masked] = -100
            r = torch.rand(ids.shape, device=device)
            mask_tok = masked & (r < 0.8)
            rand_tok = masked & (r >= 0.8) & (r < 0.9)
            ids[mask_tok] = tok.mask_token_id
            ids[rand_tok] = torch.randint(vocab, ids.shape, device=device)[rand_tok]
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item()
            n += 1
        print(f'    epoch {ep + 1}/{epochs}  train_loss {tot / max(n, 1):.4f}', flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
def model_dir(name, epochs, lr, device):
    if ARMS[name] is None:
        return BASE_MODEL
    out = RUNS / f'{name}_e{epochs}'
    if (out / 'config.json').exists():
        print(f'  reuse trained {name} at {out}')
        return str(out)
    print(f'  fine-tuning {name} ({epochs} ep) ...')
    tr = load_seqs(ARMS[name], 'train')
    finetune_mlm(tr, out, epochs, lr, device)
    return str(out)
def templates(path=''):
    if path:
        d = pd.read_csv(path)
        namecol = 'name' if 'name' in d.columns else 'uniprot_id'
        seqcol = 'seq' if 'seq' in d.columns else 'sequence' if 'sequence' in d.columns else 'wt_sequence'
        return list(zip(d[namecol], d[seqcol]))
    d = pd.read_csv(run_generation_local_ROOT / 'design' / 'design_input_proteins.csv')
    d = d[d.uniprot_id.isin(TEMPLATE_IDS)].set_index('uniprot_id').loc[TEMPLATE_IDS]
    return list(zip(d.index, d.wt_sequence))
def existing(uid):
    f = run_generation_local_GEN_DIR / f'{uid}_probs.npz'
    if not f.exists():
        return {}
    with np.load(f, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
def run_generation_local_main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=5e-05)
    ap.add_argument('--force', action='store_true', help='recompute all models')
    ap.add_argument('--phase2', action='store_true', help='also generate sequences')
    ap.add_argument('--n_designs', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--redesign', choices=['surface', 'all'], default='surface', help='surface = redesign only surface positions (WT core fixed; avoids the low-complexity collapse of full redesign).')
    ap.add_argument('--n_passes', type=int, default=2)
    ap.add_argument('--targets', default='', help='CSV with columns name/seq to design on (default: the 5 built-in templates).')
    ap.add_argument('--surface_json', default='', help='surface-positions JSON for --targets (default: generation/surface_positions.json).')
    ap.add_argument('--skip_maps', action='store_true', help='skip Phase-1 probability maps (use for a design-only run on new targets).')
    ap.add_argument('--out_tag', default='', help='tag inserted into the designs filename.')
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    run_generation_local_GEN_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f'device: {device}')
    tmpl = templates(args.targets)
    if args.skip_maps:
        need = []
        print('skipping Phase-1 maps')
    else:
        have = {uid: set(existing(uid)) for (uid, _) in tmpl}
        need = [m for m in ARMS if args.force or any((m not in have[uid] for (uid, _) in tmpl))]
        print(f"models to compute: {need or 'none (all present)'}")
    for name in need:
        path = model_dir(name, args.epochs, args.lr, device)
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()
        for (uid, seq) in tmpl:
            mat = esm_position_distributions(seq, tok, mdl, device)
            store = existing(uid)
            store.update({'seq': seq, 'aa_order': np.array(list(CANONICAL_AA)), name: mat})
            np.savez(run_generation_local_GEN_DIR / f'{uid}_probs.npz', **store)
            print(f'  {uid} {name}: {mat.shape}')
        del mdl
    design_rows = []
    if args.phase2:
        import json
        fixed_by_uid = {}
        if args.redesign == 'surface':
            surf_path = args.surface_json or run_generation_local_GEN_DIR / 'surface_positions.json'
            surf = json.load(open(surf_path))
            for (uid, seq) in tmpl:
                surf_set = set(surf[uid]['surface'])
                fixed_by_uid[uid] = [i for i in range(len(seq)) if i not in surf_set]
        for name in ARMS:
            path = model_dir(name, args.epochs, args.lr, device)
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()
            pf = make_esm_predict_fn(tok, mdl, device)
            for (uid, seq) in tmpl:
                fixed = fixed_by_uid.get(uid)
                for k in range(args.n_designs):
                    des = iterative_infill(seq, pf, n_passes=args.n_passes, temperature=args.temperature, seed=k, fixed_positions=fixed)
                    design_rows.append(dict(uniprot_id=uid, model=name, sample_idx=k, redesign=args.redesign, sequence=des))
            print(f'  generated {args.n_designs}/template with {name}')
            del mdl
    if design_rows:
        tag = f'{args.out_tag}_' if args.out_tag else ''
        out = run_generation_local_GEN_DIR / f'esm_designs_local_{tag}{args.redesign}.csv'
        df = pd.DataFrame(design_rows)
        if out.exists():
            df = pd.concat([pd.read_csv(out), df], ignore_index=True).drop_duplicates(['uniprot_id', 'model', 'sample_idx'], keep='last')
        df.to_csv(out, index=False)
        print(f'wrote {out} ({len(df)} designs)')
    print('done. Now run: python esm_design_heatmaps.py')
def run_generation_local__entry():
    sys.path.insert(0, str(run_generation_local_HERE))
    run_generation_local_main()

# ---------- from score_esm2_masked_marginals.py ----------
score_esm2_masked_marginals_CANONICAL_AA = set('ACDEFGHIKLMNPQRSTVWY')
DEFAULT_MODEL = 'facebook/esm2_t12_35M_UR50D'
def score_esm2_masked_marginals_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', default=DEFAULT_MODEL, help='HF model name or local continued-pretraining run.')
    parser.add_argument('--input_csv', help='CSV with sequences to score.')
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--id_col', default='id')
    parser.add_argument('--seq_col', default='sequence')
    parser.add_argument('--max_len', type=int, default=1022)
    parser.add_argument('--batch_masked_positions', type=int, default=64)
    parser.add_argument('--model_name', default='', help='Name to store in output; defaults to model_dir basename.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu', 'mps'])
    parser.add_argument('--limit', type=int, default=0, help='Score at most this many rows.')
    parser.add_argument('--self_test', action='store_true', help='Score three short built-in sequences.')
    return parser.parse_args()
def choose_device(device_arg: str):
    import torch
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
def read_input_rows(args: argparse.Namespace) -> List[dict]:
    if args.self_test:
        return [{args.id_col: 'selftest_1', args.seq_col: 'MKTAYIAKQRQISFVKSHFSRQ'}, {args.id_col: 'selftest_2', args.seq_col: 'GASPVTCILNDQKEMHFRYW'}, {args.id_col: 'selftest_3', args.seq_col: 'MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAEL'}]
    if not args.input_csv:
        raise SystemExit('--input_csv is required unless --self_test is set.')
    with Path(args.input_csv).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    return rows
def validate_sequence(seq: str, max_len: int) -> Optional[str]:
    if not seq:
        return 'empty_sequence'
    bad = sorted(set(seq) - score_esm2_masked_marginals_CANONICAL_AA)
    if bad:
        return 'noncanonical:' + ''.join(bad)
    if len(seq) > max_len:
        return f'length_gt_{max_len}'
    return None
def masked_marginal_score(sequence: str, tokenizer, model, device, batch_masked_positions: int) -> float:
    import torch
    encoded = tokenizer(sequence, return_tensors='pt', return_special_tokens_mask=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    special_mask = encoded['special_tokens_mask'].to(device)
    residue_positions = (special_mask[0] == 0).nonzero(as_tuple=False).flatten()
    if residue_positions.numel() != len(sequence):
        raise ValueError(f'Residue-token mismatch: {residue_positions.numel()} non-special tokens for {len(sequence)} residues.')
    if tokenizer.mask_token_id is None:
        raise ValueError('Tokenizer has no mask token.')
    true_token_ids = input_ids[0, residue_positions]
    log_probs: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequence), batch_masked_positions):
            end = min(start + batch_masked_positions, len(sequence))
            chunk_positions = residue_positions[start:end]
            batch = input_ids.repeat(chunk_positions.numel(), 1)
            batch_attention = attention_mask.repeat(chunk_positions.numel(), 1)
            row_idx = torch.arange(chunk_positions.numel(), device=device)
            batch[row_idx, chunk_positions] = tokenizer.mask_token_id
            outputs = model(input_ids=batch, attention_mask=batch_attention)
            logits = outputs.logits[row_idx, chunk_positions, :]
            chunk_log_probs = torch.log_softmax(logits, dim=-1)
            true_ids = true_token_ids[start:end]
            gathered = chunk_log_probs.gather(1, true_ids.unsqueeze(1)).squeeze(1)
            log_probs.extend(gathered.detach().cpu().tolist())
    if len(log_probs) != len(sequence):
        raise RuntimeError(f'Expected {len(sequence)} log probabilities, got {len(log_probs)}.')
    score = float(sum(log_probs) / len(log_probs))
    if not math.isfinite(score):
        raise RuntimeError('Non-finite masked-marginal score.')
    return score
def score_esm2_masked_marginals_main() -> None:
    args = score_esm2_masked_marginals_parse_args()
    rows = read_input_rows(args)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir).to(device)
    model_name = args.model_name or (args.model_dir if '/' in args.model_dir else Path(args.model_dir).name)
    output_rows = []
    for (idx, row) in enumerate(rows, start=1):
        seq = (row.get(args.seq_col) or '').strip().upper()
        seq_id = row.get(args.id_col) or f'row_{idx}'
        status = validate_sequence(seq, args.max_len)
        score = ''
        if status is None:
            try:
                score = masked_marginal_score(seq, tokenizer, model, device, args.batch_masked_positions)
                status = 'ok'
            except Exception as exc:
                status = f'error:{type(exc).__name__}:{exc}'
                score = ''
        output_rows.append({'id': seq_id, 'sequence_length': len(seq), 'esm_mlm_score': score, 'status': status, 'model_name': model_name})
        print(f'[{idx}/{len(rows)}] {seq_id} length={len(seq)} status={status}', flush=True)
    with out_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'sequence_length', 'esm_mlm_score', 'status', 'model_name'])
        writer.writeheader()
        writer.writerows(output_rows)
    ok_scores = [float(row['esm_mlm_score']) for row in output_rows if row['status'] == 'ok']
    if ok_scores:
        print(f'Wrote {out_path}; ok={len(ok_scores)} min={min(ok_scores):.4f} median={sorted(ok_scores)[len(ok_scores) // 2]:.4f} max={max(ok_scores):.4f}')
    else:
        print(f'Wrote {out_path}; no successful scores.')
def score_esm2_masked_marginals__entry():
    score_esm2_masked_marginals_main()

# ---------- from analyse_esm2_score_shifts.py ----------
DEFAULT_FEATURE_TABLE = 'dataset_update/main_plus_r2_r3_analysis_v12_cli.csv'
FEATURES = ['acidic_residue_fraction', 'basic_residue_fraction', 'charge_per_residue', 'isoelectric_point']
MODEL_COHORT = {'AlkSecESM35M': 'alkaliphile secretome', 'AcidSecESM35M': 'acidophile secretome', 'NeuSecESM35M_AlkMatched': 'alkaliphile-matched neutralophile secretome', 'NeuSecESM35M_AcidMatched': 'acidophile-matched neutralophile secretome'}
PROTEINMPNN_COLUMNS = {'AlkSecMPNN_020': 'AlkSecMPNN_020_score', 'AcidSecMPNN_020': 'AcidSecMPNN_020_score', 'AlkSecMPNN': 'AlkSecMPNN_v2_score', 'AcidSecMPNN': 'AcidSecMPNN_score'}
def parse_model_score(value: str) -> Tuple[str, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('--continued_score must be ModelName=/path/to/scores.csv')
    (name, path) = value.split('=', 1)
    if not name or not path:
        raise argparse.ArgumentTypeError('--continued_score must be ModelName=/path/to/scores.csv')
    return (name, Path(path))
def analyse_esm2_score_shifts_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_scores', required=True)
    parser.add_argument('--continued_score', action='append', type=parse_model_score, required=True, help='Repeated ModelName=/path/to/score.csv argument.')
    parser.add_argument('--feature_table', default=DEFAULT_FEATURE_TABLE, help='Feature table with acid-base features, or a sequence table from which they can be computed.')
    parser.add_argument('--out_dir', default='outputs/esm35m_continual_pretraining')
    parser.add_argument('--score_id_col', default='id')
    parser.add_argument('--feature_id_col', default='auto')
    parser.add_argument('--feature_seq_col', default='sequence')
    parser.add_argument('--proteinmpnn_score_table', default='', help='Defaults to feature_table when possible.')
    parser.add_argument('--proteinmpnn_base_col', default='ProteinMPNN_v020_score')
    return parser.parse_args()
def read_csv(path: Path) -> List[dict]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))
def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
def to_float(value) -> float:
    if value in (None, ''):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
def choose_feature_id_col(rows: List[dict], requested: str) -> str:
    if requested != 'auto':
        return requested
    if not rows:
        return 'id'
    columns = set(rows[0])
    for candidate in ('id', 'Entry', 'acc', 'name'):
        if candidate in columns:
            return candidate
    raise ValueError(f'Could not infer feature ID column from columns: {sorted(columns)[:20]}')
def net_charge_at_ph(seq: str, ph: float) -> float:
    pka_pos = {'K': 10.5, 'R': 12.4, 'H': 6.0}
    pka_neg = {'D': 3.9, 'E': 4.1, 'C': 8.3, 'Y': 10.1}
    n_term = 9.69
    c_term = 2.34
    charge = 1.0 / (1.0 + 10.0 ** (ph - n_term))
    charge -= 1.0 / (1.0 + 10.0 ** (c_term - ph))
    for (aa, pka) in pka_pos.items():
        charge += seq.count(aa) / (1.0 + 10.0 ** (ph - pka))
    for (aa, pka) in pka_neg.items():
        charge -= seq.count(aa) / (1.0 + 10.0 ** (pka - ph))
    return charge
def isoelectric_point(seq: str) -> float:
    (low, high) = (0.0, 14.0)
    for _ in range(80):
        mid = (low + high) / 2.0
        if net_charge_at_ph(seq, mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0
def add_features_from_sequence(rows: List[dict], seq_col: str) -> None:
    for row in rows:
        if all((row.get(feature) not in (None, '') for feature in FEATURES)):
            continue
        seq = (row.get(seq_col) or '').strip().upper()
        n = len(seq)
        if not n:
            continue
        row['acidic_residue_fraction'] = row.get('acidic_residue_fraction') or (seq.count('D') + seq.count('E')) / n
        row['basic_residue_fraction'] = row.get('basic_residue_fraction') or (seq.count('K') + seq.count('R') + seq.count('H')) / n
        row['charge_per_residue'] = row.get('charge_per_residue') or net_charge_at_ph(seq, 7.0) / n
        row['isoelectric_point'] = row.get('isoelectric_point') or isoelectric_point(seq)
def read_scores(path: Path, id_col: str) -> Dict[str, float]:
    rows = read_csv(path)
    scores: Dict[str, float] = {}
    for row in rows:
        if row.get('status', 'ok') != 'ok':
            continue
        score = to_float(row.get('esm_mlm_score'))
        if math.isfinite(score):
            scores[row[id_col]] = score
    return scores
def zscore(values):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr, ddof=0)
    if not math.isfinite(std) or std == 0:
        return arr * 0.0
    return (arr - mean) / std
def spearman(x, y) -> Tuple[float, float]:
    import numpy as np
    from scipy.stats import spearmanr
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (math.nan, math.nan)
    (rho, p) = spearmanr(x[mask], y[mask])
    return (float(rho), float(p))
def acid_base_pca(feature_matrix):
    import numpy as np
    from sklearn.decomposition import PCA
    x = np.asarray(feature_matrix, dtype=float)
    xz = np.column_stack([zscore(x[:, i]) for i in range(x.shape[1])])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(xz)
    loading = pca.components_[0]
    acid_orientation = loading[0] - loading[1] - loading[2] - loading[3]
    if acid_orientation < 0:
        pcs[:, 0] *= -1.0
        pca.components_[0] *= -1.0
    return (pcs, pca)
def ols_pc_model(delta_scores, pcs) -> Dict[str, float]:
    import numpy as np
    from scipy.stats import t as t_dist
    y = zscore(delta_scores)
    x1 = zscore(pcs[:, 0])
    x2 = zscore(pcs[:, 1])
    X = np.column_stack([np.ones_like(y), x1, x2])
    (beta, *_) = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else math.nan
    df = len(y) - X.shape[1]
    pvals = [math.nan, math.nan, math.nan]
    if df > 0:
        sigma2 = ss_res / df
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        with np.errstate(divide='ignore', invalid='ignore'):
            tvals = beta / se
        pvals = [float(2.0 * t_dist.sf(abs(t), df)) if math.isfinite(float(t)) else math.nan for t in tvals]
    return {'grad_PC1': float(beta[1]), 'grad_PC2': float(beta[2]), 'grad_PC1_pvalue': pvals[1], 'grad_PC2_pvalue': pvals[2], 'R2': r2}
def assemble_analysis_rows(base_scores: Dict[str, float], continued_scores: Dict[str, float], feature_rows: Dict[str, dict]):
    import numpy as np
    ids = sorted(set(base_scores) & set(continued_scores) & set(feature_rows))
    rows = []
    for seq_id in ids:
        features = [to_float(feature_rows[seq_id].get(feature)) for feature in FEATURES]
        if not all((math.isfinite(value) for value in features)):
            continue
        rows.append({'id': seq_id, 'delta_score': continued_scores[seq_id] - base_scores[seq_id], **{feature: value for (feature, value) in zip(FEATURES, features)}})
    return rows
def analyse_delta(model: str, rows: List[dict]) -> Tuple[dict, List[dict]]:
    import numpy as np
    delta = np.asarray([row['delta_score'] for row in rows], dtype=float)
    feat = np.asarray([[row[feature] for feature in FEATURES] for row in rows], dtype=float)
    (pcs, _) = acid_base_pca(feat)
    ols = ols_pc_model(delta, pcs)
    corr_map = {}
    direct_rows = []
    for feature in FEATURES:
        (rho, p) = spearman(delta, np.asarray([row[feature] for row in rows], dtype=float))
        suffix = {'acidic_residue_fraction': 'acidic', 'basic_residue_fraction': 'basic', 'charge_per_residue': 'charge', 'isoelectric_point': 'pI'}[feature]
        corr_map[f'corr_{suffix}'] = rho
        corr_map[f'corr_{suffix}_pvalue'] = p
        direct_rows.append({'model': model, 'training_cohort': MODEL_COHORT.get(model, ''), 'feature': feature, 'spearman_rho': rho, 'pvalue': p, 'n': len(rows)})
    summary = {'model': model, 'training_cohort': MODEL_COHORT.get(model, ''), 'n': len(rows), **ols, **corr_map}
    return (summary, direct_rows)
def plot_summary(summary_rows: List[dict], direct_rows: List[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = [row['model'] for row in summary_rows]
    values = [float(row['grad_PC1']) for row in summary_rows]
    colors = ['#b44d4d' if 'Alk' in m else '#4d76b4' if 'Acid' in m else '#7a7a7a' for m in models]
    (fig, ax) = plt.subplots(figsize=(max(6, 1.5 * len(models)), 4))
    ax.bar(models, values, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('gradient on acidic PC1')
    ax.set_title('Delta score = continued-pretrained ESM2-35M score - base ESM2-35M score')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / 'Fig_ESM35M_dscore_acidbase_PC1.png', dpi=220)
    plt.close(fig)
    features = FEATURES
    matrix = []
    for model in models:
        model_rows = {row['feature']: row for row in direct_rows if row['model'] == model}
        matrix.append([float(model_rows[feature]['spearman_rho']) for feature in features])
    (fig, ax) = plt.subplots(figsize=(8, max(3, 0.7 * len(models))))
    im = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(features)), labels=['acidic', 'basic', 'charge', 'pI'])
    ax.set_yticks(range(len(models)), labels=models)
    ax.set_title('Spearman correlations with delta score')
    for i in range(len(models)):
        for j in range(len(features)):
            ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, label='Spearman rho')
    fig.tight_layout()
    fig.savefig(fig_dir / 'Fig_ESM35M_direct_correlations.png', dpi=220)
    plt.close(fig)
def proteinmpnn_comparison_rows(score_table: Path, base_col: str) -> List[dict]:
    rows = read_csv(score_table)
    if not rows or base_col not in rows[0]:
        return []
    feature_id_col = choose_feature_id_col(rows, 'auto')
    add_features_from_sequence(rows, 'sequence')
    by_id = {row[feature_id_col]: row for row in rows}
    out = []
    for (model, ft_col) in PROTEINMPNN_COLUMNS.items():
        if ft_col not in rows[0]:
            continue
        analysis_rows = []
        for row in rows:
            base = to_float(row.get(base_col))
            ft = to_float(row.get(ft_col))
            feats = [to_float(row.get(feature)) for feature in FEATURES]
            if math.isfinite(base) and math.isfinite(ft) and all((math.isfinite(v) for v in feats)):
                analysis_rows.append({'id': row[feature_id_col], 'delta_score': ft - base, **{feature: value for (feature, value) in zip(FEATURES, feats)}})
        if len(analysis_rows) < 4:
            continue
        (summary, _) = analyse_delta(model, analysis_rows)
        out.append({'model_family': 'ProteinMPNN', 'model': model, 'training_cohort': 'alkaliphile/acidophile secretome', 'readout': 'WT sequence score shift', 'grad_PC1': summary['grad_PC1'], 'R2': summary['R2'], 'corr_acidic': summary['corr_acidic'], 'corr_charge': summary['corr_charge'], 'corr_pI': summary['corr_pI'], 'interpretation': 'Also has a structure-conditioned design-generation endpoint; WT-score shift is secondary.'})
    return out
def analyse_esm2_score_shifts_main() -> None:
    args = analyse_esm2_score_shifts_parse_args()
    out_dir = Path(args.out_dir)
    table_dir = out_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    feature_table = Path(args.feature_table)
    feature_rows_list = read_csv(feature_table)
    add_features_from_sequence(feature_rows_list, args.feature_seq_col)
    feature_id_col = choose_feature_id_col(feature_rows_list, args.feature_id_col)
    feature_rows = {row[feature_id_col]: row for row in feature_rows_list}
    print(f'Using feature table: {feature_table} (id column: {feature_id_col})')
    base_scores = read_scores(Path(args.base_scores), args.score_id_col)
    summary_rows = []
    direct_rows = []
    comparison_rows = []
    for (model, score_path) in args.continued_score:
        continued_scores = read_scores(score_path, args.score_id_col)
        rows = assemble_analysis_rows(base_scores, continued_scores, feature_rows)
        if len(rows) < 4:
            raise RuntimeError(f'Too few overlapping scored/feature rows for {model}: n={len(rows)}')
        (summary, direct) = analyse_delta(model, rows)
        summary_rows.append(summary)
        direct_rows.extend(direct)
        comparison_rows.append({'model_family': 'ESM2-35M', 'model': model, 'training_cohort': summary['training_cohort'], 'readout': 'WT sequence masked-marginal score shift', 'grad_PC1': summary['grad_PC1'], 'R2': summary['R2'], 'corr_acidic': summary['corr_acidic'], 'corr_charge': summary['corr_charge'], 'corr_pI': summary['corr_pI'], 'interpretation': 'Sequence-only model; only WT-score reranking is assessed.'})
    summary_fields = ['model', 'training_cohort', 'n', 'grad_PC1', 'grad_PC1_pvalue', 'grad_PC2', 'grad_PC2_pvalue', 'R2', 'corr_acidic', 'corr_acidic_pvalue', 'corr_basic', 'corr_basic_pvalue', 'corr_charge', 'corr_charge_pvalue', 'corr_pI', 'corr_pI_pvalue']
    write_csv(table_dir / 'Table_ESM35M_score_shift_summary.csv', summary_rows, summary_fields)
    write_csv(table_dir / 'Table_ESM35M_direct_correlations.csv', direct_rows, ['model', 'training_cohort', 'feature', 'spearman_rho', 'pvalue', 'n'])
    proteinmpnn_table = Path(args.proteinmpnn_score_table) if args.proteinmpnn_score_table else feature_table
    comparison_rows.extend(proteinmpnn_comparison_rows(proteinmpnn_table, args.proteinmpnn_base_col))
    write_csv(table_dir / 'Table_sequence_vs_structure_score_shift_comparison.csv', comparison_rows, ['model_family', 'model', 'training_cohort', 'readout', 'grad_PC1', 'R2', 'corr_acidic', 'corr_charge', 'corr_pI', 'interpretation'])
    plot_summary(summary_rows, direct_rows, out_dir)
    print(f'Wrote analysis tables and figures under {out_dir}')
def analyse_esm2_score_shifts__entry():
    analyse_esm2_score_shifts_main()

# ---------- from esm_design_surface.py ----------
esm_design_surface_HERE = Path(__file__).resolve().parent
esm_design_surface_ROOT = esm_design_surface_HERE.parents[1]
GEN = esm_design_surface_ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'generation'
def surface_letters(seq, surf_idx):
    return [seq[i] for i in surf_idx if i < len(seq) and seq[i] in MAXASA]
def complexity(seq):
    """Shannon entropy (bits) of the AA composition - low = collapsed."""
    from collections import Counter
    n = len(seq)
    p = np.array([v / n for v in Counter(seq).values()])
    return float(-(p * np.log2(p)).sum())
def esm_design_surface_main():
    src = GEN / 'esm_designs_local_surface.csv'
    if not src.exists():
        src = GEN / 'esm_designs_local.csv'
    print(f'designs: {src.name}')
    designs = pd.read_csv(src)
    inp = pd.read_csv(esm_design_surface_ROOT / 'design' / 'design_input_proteins.csv').set_index('uniprot_id')
    rows = []
    for (uid, g) in designs.groupby('uniprot_id'):
        path = inp.loc[uid, 'structure_pdb_v6']
        (letters, rsa) = per_residue_rsa(path)
        surf_idx = [i for (i, r) in enumerate(rsa) if not np.isnan(r) and r >= RSA_CUT]
        wt = inp.loc[uid, 'wt_sequence']
        wt_net = comp(surface_letters(wt, surf_idx)).get('net_KR_DE', np.nan)
        for (model, gm) in g.groupby('model'):
            nets = [comp(surface_letters(s, surf_idx)).get('net_KR_DE', np.nan) for s in gm.sequence]
            ents = [complexity(s) for s in gm.sequence]
            rows.append(dict(uniprot_id=uid, model=model, surf_net=np.nanmean(nets), surf_net_shift=np.nanmean(nets) - wt_net, mean_entropy=np.mean(ents), wt_surf_net=wt_net))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index='uniprot_id', columns='model', values='surf_net')
    ent = df.pivot_table(index='uniprot_id', columns='model', values='mean_entropy')
    print('=== surface net charge (K+R-D-E)/n_surf of designs ===')
    print(piv.round(3).to_string())
    print('\n=== WT surface net charge ===')
    print(df.groupby('uniprot_id').wt_surf_net.first().round(3).to_string())
    print('\n=== design sequence entropy (bits; WT-like ~4; <2 = collapsed) ===')
    print(ent.round(2).to_string())
    df.to_csv(GEN / 'esm_design_surface_summary.csv', index=False)
    print('\nwrote esm_design_surface_summary.csv')
def esm_design_surface__entry():
    sys.path.insert(0, str(esm_design_surface_ROOT / 'design'))
    esm_design_surface_main()

# ---------- from esm_design_heatmaps.py ----------
esm_design_heatmaps_HERE = Path(__file__).resolve().parent
esm_design_heatmaps_ROOT = esm_design_heatmaps_HERE.parents[1]
esm_design_heatmaps_GEN_DIR = esm_design_heatmaps_ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'generation'
FT_KEY = 'AlkSecESM35M'
CTRL_KEY = 'NeuSecESM35M'
esm_design_heatmaps_RSA_CUT = 0.25
def structure_paths():
    d = pd.read_csv(esm_design_heatmaps_ROOT / 'design' / 'design_input_proteins.csv')
    return dict(zip(d.uniprot_id, d.structure_pdb_v6))
def surface_mask(uid, seq, paths):
    """Boolean per-position surface mask (RSA>=0.25); None if unavailable."""
    p = paths.get(uid)
    if not p or not Path(p).exists():
        return None
    try:
        from surface_features_alkaline import per_residue_rsa
        (_, rsa) = per_residue_rsa(p)
    except Exception as e:
        print(f'  {uid}: surface RSA unavailable ({e})')
        return None
    if len(rsa) != len(seq):
        print(f'  {uid}: RSA length {len(rsa)} != seq {len(seq)}; skipping surface overlay')
        return None
    return np.nan_to_num(rsa, nan=0.0) >= esm_design_heatmaps_RSA_CUT
def plot_template(uid, npz, paths, out_dir):
    seq = str(npz['seq'])
    aa = ''.join(npz['aa_order'].tolist()) if 'aa_order' in npz else CANONICAL_AA
    (base, ft) = (npz['base'], npz[FT_KEY])
    delta = ft - base
    prop_base = acidic_propensity(base, aa)
    prop_ft = acidic_propensity(ft, aa)
    dprop = prop_ft - prop_base
    has_ctrl = CTRL_KEY in npz.files
    if has_ctrl:
        prop_ctrl = acidic_propensity(npz[CTRL_KEY], aa)
        dprop_ctrl = prop_ctrl - prop_base
    surf = surface_mask(uid, seq, paths)
    (fig, (ax1, ax2)) = plt.subplots(2, 1, figsize=(max(7, len(seq) * 0.12), 6.5), gridspec_kw={'height_ratios': [3, 1.4]}, sharex=True)
    vmax = np.abs(delta).max() or 1e-06
    im = ax1.imshow(delta.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_yticks(range(len(aa)))
    ax1.set_yticklabels(list(aa), fontsize=7)
    for (i, a) in enumerate(aa):
        if a in ACIDIC:
            ax1.get_yticklabels()[i].set_color('#B2182B')
        if a in BASIC:
            ax1.get_yticklabels()[i].set_color('#2166AC')
    ax1.set_ylabel('amino acid')
    ax1.set_title(f'{uid}  ({len(seq)} aa)   FT - base per-position probability  (red=FT prefers more)')
    fig.colorbar(im, ax=ax1, fraction=0.02, pad=0.01, label='Δ prob')
    ax2.axhline(0, color='grey', lw=0.6)
    ax2.plot(prop_base, color='grey', lw=1, label='base')
    ax2.plot(prop_ft, color='#B2182B', lw=1, label='AlkSec FT')
    if has_ctrl:
        ax2.plot(prop_ctrl, color='#2166AC', lw=1, ls='--', label='NeuSec ctrl')
    ax2.set_ylabel('acidic\npropensity', fontsize=8)
    ax2.set_xlabel('position')
    ax2.legend(fontsize=7, loc='upper right')
    if surf is not None:
        for x in np.where(surf)[0]:
            ax2.axvspan(x - 0.5, x + 0.5, color='#f4d03f', alpha=0.18, lw=0)
        ax2.text(0.01, 0.02, 'shaded = surface (RSA≥0.25)', transform=ax2.transAxes, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_dir / f'fig_esm_heatmap_{uid}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    row = {'uniprot_id': uid, 'length': len(seq), 'alk_dprop_all': float(np.mean(dprop))}
    if has_ctrl:
        row['neu_dprop_all'] = float(np.mean(dprop_ctrl))
    if surf is not None and surf.any():
        core = ~surf
        row['alk_dprop_surface'] = float(np.mean(dprop[surf]))
        row['alk_dprop_core'] = float(np.mean(dprop[core])) if core.any() else np.nan
        if has_ctrl:
            row['neu_dprop_surface'] = float(np.mean(dprop_ctrl[surf]))
            row['surface_specificity'] = row['alk_dprop_surface'] - row['neu_dprop_surface']
        row['n_surface'] = int(surf.sum())
    return row
def esm_design_heatmaps_main():
    out_dir = esm_design_heatmaps_GEN_DIR
    npz_files = sorted(esm_design_heatmaps_GEN_DIR.glob('*_probs.npz'))
    if not npz_files:
        print(f'No *_probs.npz in {esm_design_heatmaps_GEN_DIR}. Run the Phase-1 Colab notebook and unzip here.')
        return
    paths = structure_paths()
    rows = []
    for f in npz_files:
        uid = f.name.replace('_probs.npz', '')
        with np.load(f, allow_pickle=True) as npz:
            rows.append(plot_template(uid, npz, paths, out_dir))
        print(f'  wrote fig_esm_heatmap_{uid}.png')
    summ = pd.DataFrame(rows)
    summ.to_csv(esm_design_heatmaps_GEN_DIR / 'esm_design_heatmap_summary.csv', index=False)
    print('\n=== Δ acidic-propensity (FT - base): >0 means FT prefers MORE acidic ===')
    print(summ.round(4).to_string(index=False))
def esm_design_heatmaps__entry():
    matplotlib.use('Agg')
    sys.path.insert(0, str(esm_design_heatmaps_HERE))
    sys.path.insert(0, str(esm_design_heatmaps_ROOT / 'design'))
    esm_design_heatmaps_main()

_STEPS = {
    'run-generation-local': run_generation_local__entry,
    'score-esm2-masked-marginals': score_esm2_masked_marginals__entry,
    'analyse-esm2-score-shifts': analyse_esm2_score_shifts__entry,
    'esm-design-surface': esm_design_surface__entry,
    'esm-design-heatmaps': esm_design_heatmaps__entry,
}

def main(argv=None):
    import sys as _sys
    argv=_sys.argv if argv is None else argv
    if len(argv)<2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    _sys.argv=[argv[0]]+argv[2:]; _STEPS[argv[1]](); return 0

if __name__=='__main__':
    raise SystemExit(main())

