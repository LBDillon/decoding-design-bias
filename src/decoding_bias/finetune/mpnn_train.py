"""decoding_bias.finetune.mpnn_train

Merged provenance module. Sections (see ARCHIVE_MAP.md):
  - train 
  - sweeps 
  - evaluate 
  - run_proteinmpnn_surface 
  - ft020_base_vs_ft_self_consistency 
  - ft020_self_consistency_vs_afdb 
"""

import argparse
import design_common as dc
import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import shutil
import subprocess
import sys
import time
import torch
import utils as U
import warnings
from pathlib import Path
from scipy.stats import wilcoxon
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data

# ---------- from train.py ----------
train_HERE = Path(__file__).resolve().parent
train_FT = train_HERE.parent
train_ROOT = train_FT.parent
JSONL = train_FT / 'data' if (train_FT / 'data').exists() else train_ROOT / 'design' / 'outputs'
def load_records(jsonl_dir, split, role, exclude_clade='', label='alkaliphile'):
    recs = []
    for line in open(Path(jsonl_dir) / f'{label}_parsed_{split}.jsonl'):
        r = json.loads(line)
        if role and r.get('role') != role:
            continue
        if exclude_clade and r.get('clade') == exclude_clade:
            continue
        r['masked_list'] = ['A']
        r['visible_list'] = []
        recs.append(r)
    return recs
def batches(recs, batch_tokens):
    recs = sorted(recs, key=lambda r: len(r['seq']))
    (out, cur, clen) = ([], [], 0)
    for r in recs:
        L = len(r['seq'])
        if cur and (clen + L > batch_tokens or len(cur) >= 8):
            out.append(cur)
            (cur, clen) = ([], 0)
        cur.append(r)
        clen += L
    if cur:
        out.append(cur)
    return out
def train_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mpnn', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--role', required=True, choices=['case', 'control'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--batch_tokens', type=int, default=3000)
    ap.add_argument('--backbone_noise', type=float, default=0.2)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--jsonl_dir', default=str(JSONL), help='dir with alkaliphile_parsed_*.jsonl')
    ap.add_argument('--out_root', default=str(train_FT / 'outputs'), help='output root dir')
    ap.add_argument('--save_every', type=int, default=0, help='also save epoch_NN.pt every N epochs')
    ap.add_argument('--warmup_steps', type=int, default=0, help='linear LR warmup over N gradient steps (0 = constant lr, the locked default). Use to test whether the 1-epoch steering is a no-warmup artefact.')
    ap.add_argument('--freeze', default='none', choices=['none', 'wout'], help='wout = freeze all but the output projection W_out (readout-only probe, steelman #2)')
    ap.add_argument('--schedule', default='const', choices=['const', 'noam'], help='const = Adam at --lr (locked recipe); noam = get_std_opt warmup (original ProteinMPNN schedule, comparable to the acidophilic baseline). noam ignores --lr/--warmup_steps and ramps lr gently.')
    ap.add_argument('--cohort', default='alkaline', choices=['alkaline', 'acid'], help="which cohort's *_parsed_*.jsonl to train on (alkaline=alkaliphile, acid=acidophile)")
    ap.add_argument('--exclude_clade', default='', help='drop this clade from training cases (leave-one-clade-out)')
    ap.add_argument('--frac', type=float, default=1.0, help='use this fraction of training cases (dose-response)')
    a = ap.parse_args()
    sys.path.insert(0, a.mpnn)
    sys.path.insert(0, str(Path(a.mpnn) / 'training'))
    from model_utils import featurize, get_std_opt
    from protein_mpnn_utils import ProteinMPNN, loss_smoothed
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    label = {'alkaline': 'alkaliphile', 'acid': 'acidophile'}[a.cohort]
    tr = load_records(a.jsonl_dir, 'train', a.role, a.exclude_clade, label)
    va = load_records(a.jsonl_dir, 'val', a.role, a.exclude_clade, label)
    if a.exclude_clade:
        print(f'[exclude_clade] dropped {a.exclude_clade}; {len(tr)} train cases remain')
    if a.frac < 1.0:
        random.shuffle(tr)
        tr = tr[:max(1, int(len(tr) * a.frac))]
        print(f'[frac] using {a.frac:.0%} of train = {len(tr)} cases')
    if a.smoke:
        tr = tr[:a.smoke]
        va = va[:max(2, a.smoke // 4)]
    print(f'[{a.run}] train {len(tr)} | val {len(va)} ({a.role}); device {device}')
    (tr_b, va_b) = (batches(tr, a.batch_tokens), batches(va, a.batch_tokens))
    ckpt = torch.load(Path(a.mpnn) / 'vanilla_model_weights' / 'v_48_002.pt', map_location=device)
    model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, k_neighbors=ckpt['num_edges'], augment_eps=a.backbone_noise, dropout=a.dropout)
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    if a.freeze == 'wout':
        for (nm, p) in model.named_parameters():
            p.requires_grad = nm.startswith('W_out')
        print(f'[freeze] training W_out only: {sum((p.numel() for p in model.parameters() if p.requires_grad))} params')
    train_params = [p for p in model.parameters() if p.requires_grad]
    if a.schedule == 'noam':
        opt = get_std_opt(train_params, 128, 0)
        print('[opt] Noam warmup (get_std_opt: d_model=128, factor=2, warmup=4000) - lr ramps gently')
    else:
        opt = torch.optim.Adam(train_params, lr=a.lr, betas=(0.9, 0.98), eps=1e-09)
    gstep = 0

    def lr_for(step):
        return a.lr * (step + 1) / a.warmup_steps if a.warmup_steps and step < a.warmup_steps else a.lr

    def epoch(bs, train):
        nonlocal gstep
        model.train(train)
        tot = acc = res = 0.0
        if train:
            random.shuffle(bs)
        for batch in bs:
            (X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_enc) = featurize(batch, device)
            mfl = mask * chain_M
            randn = torch.randn(chain_M.shape, device=device)
            if train:
                opt.zero_grad()
            with torch.set_grad_enabled(train):
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_enc, randn)
                (_, loss_av) = loss_smoothed(S, log_probs, mfl)
                if train:
                    loss_av.backward()
                    torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                    if a.schedule == 'const':
                        for g in opt.param_groups:
                            g['lr'] = lr_for(gstep)
                    opt.step()
                    gstep += 1
            nll = -log_probs.gather(-1, S.unsqueeze(-1)).squeeze(-1)
            tf = (log_probs.argmax(-1) == S).float()
            w = mfl.sum().item()
            tot += (nll * mfl).sum().item()
            acc += (tf * mfl).sum().item()
            res += w
        return (tot / res, acc / res)
    od = Path(a.out_root) / a.run
    (od / 'model_weights').mkdir(parents=True, exist_ok=True)
    (od / 'log.txt').write_text('')
    json.dump(vars(a), open(od / 'config.json', 'w'), indent=2)
    best = 1000000000.0
    for ep in range(a.epochs):
        t0 = time.time()
        (trn, tra) = epoch(tr_b, True)
        with torch.no_grad():
            (vn, vaa) = epoch(va_b, False)
        line = f"epoch {ep} train_nll {trn:.3f} train_rec {tra:.3f} | val_nll {vn:.3f} val_rec {vaa:.3f} | lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s"
        print(line, flush=True)
        open(od / 'log.txt', 'a').write(line + '\n')
        save = {'model_state_dict': model.state_dict(), 'num_edges': ckpt['num_edges'], 'epoch': ep, 'step': 0}
        torch.save(save, od / 'model_weights' / 'epoch_last.pt')
        if vn < best:
            best = vn
            torch.save(save, od / 'model_weights' / 'epoch_best.pt')
        if a.save_every and (ep % a.save_every == 0 or ep == a.epochs - 1):
            torch.save(save, od / 'model_weights' / f'epoch_{ep:02d}.pt')
    print(f'[{a.run}] done. best val_nll {best:.3f}')
def train__entry():
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    warnings.filterwarnings('ignore')
    train_main()

# ---------- from sweeps.py ----------
def eval_model(model, bb, n, temp, device):
    rows = [U.per_backbone(model, pdb, n, temp, device) for (_, _, pdb) in bb]
    return (np.mean([r['surface_net'] for r in rows]), np.mean([r['recovery'] for r in rows]))
def checkpoint_sweep(a, device, bb, gap):
    (base_sn, base_rec) = eval_model(U.load_model(a.base, device), bb, a.n, 0.1, device)
    print(f'base: surface_net {base_sn:.4f} recovery {base_rec:.4f}\n')
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, 'epoch_[0-9]*.pt')), key=lambda p: int(re.search('epoch_(\\d+)', p).group(1)))
    print(f"{'epoch':>6}{'surf_net':>10}{'shift':>9}{'%gap':>7}{'recovery':>10}{'drop_pp':>9}{'guardrail':>10}")
    (passing, rows) = ([], [{'epoch': -1, 'surf_net': base_sn, 'pct_gap': 0.0, 'recovery': base_rec, 'drop_pp': 0.0, 'pass': 'base'}])
    for c in ckpts:
        ep = int(re.search('epoch_(\\d+)', c).group(1))
        (sn, rec) = eval_model(U.load_model(c, device), bb, a.n, 0.1, device)
        shift = sn - base_sn
        fgap = -shift / gap['surface_net']
        drop = base_rec - rec
        ok = drop <= a.guardrail and fgap >= a.min_gap
        if ok:
            passing.append(ep)
        rows.append({'epoch': ep, 'surf_net': sn, 'pct_gap': 100 * fgap, 'recovery': rec, 'drop_pp': 100 * drop, 'pass': ok})
        print(f"{ep:>6}{sn:>10.4f}{shift:>9.4f}{100 * fgap:>6.0f}%{rec:>10.4f}{100 * drop:>8.1f}{('PASS' if ok else '-'):>10}")
    if a.save_csv:
        pd.DataFrame(rows).to_csv(a.save_csv, index=False)
        print(f'[saved] {a.save_csv}')
    safe = [r for r in rows if r['epoch'] >= 0 and r['drop_pp'] <= 100 * a.guardrail]
    if a.select == 'maxsteer':
        chosen = max(safe, key=lambda r: r['epoch']) if safe else None
    else:
        elig = [r for r in safe if r['pct_gap'] >= 100 * a.min_gap]
        chosen = min(elig, key=lambda r: r['epoch']) if elig else None
    if chosen:
        print(f"\nSELECTED (rule={a.select}, guardrail <= {100 * a.guardrail:.0f}pp): epoch_{int(chosen['epoch']):02d}  ({chosen['pct_gap']:.0f}% of gap, {chosen['drop_pp']:.1f}pp recovery drop)")
        if a.select_out:
            Path(a.select_out).write_text(f"{int(chosen['epoch']):02d}")
            print(f'[selected -> {a.select_out}]')
    else:
        print('\nNo checkpoint within the recovery guardrail; report the tradeoff curve.')
def temperature_sweep(a, device, bb, gap):
    base = U.load_model(a.base, device)
    ft = U.load_model(a.ckpt, device)
    print(f"{'T':>5}{'base_sn':>10}{'ft_sn':>10}{'shift':>9}{'%gap':>7}{'base_rec':>9}{'ft_rec':>8}")
    rows = []
    for T in [float(t) for t in a.temps.split(',')]:
        (bsn, brec) = eval_model(base, bb, a.n, T, device)
        (fsn, frec) = eval_model(ft, bb, a.n, T, device)
        sh = fsn - bsn
        rows.append({'T': T, 'pct_gap': -sh / gap['surface_net'] * 100, 'base_rec': brec, 'ft_rec': frec, 'shift': sh})
        print(f"{T:>5}{bsn:>10.4f}{fsn:>10.4f}{sh:>9.4f}{-sh / gap['surface_net'] * 100:>6.0f}%{brec:>9.3f}{frec:>8.3f}", flush=True)
    if a.save_csv:
        pd.DataFrame(rows).to_csv(a.save_csv, index=False)
        print(f'[saved] {a.save_csv}')
def logitbias_sweep(a, device, bb, gap):
    """Hand-coded logit bias toward acidic (D,E) / away from basic (K,R) on the BASE model - the
    'why not just bias the logits?' baseline. Compare its surface_net-vs-recovery tradeoff to FT_alk."""
    base = U.load_model(a.base, device)
    ALPH = 'ACDEFGHIKLMNPQRSTVWYX'

    def vec(b):
        v = np.zeros(21, np.float32)
        for x in 'DE':
            v[ALPH.index(x)] = b
        for x in 'KR':
            v[ALPH.index(x)] = -b
        return v
    (base_sn, rows) = (None, [])
    print(f"{'bias':>6}{'surf_net':>10}{'shift':>9}{'%gap':>7}{'recovery':>10}")
    for b in [float(x) for x in a.biases.split(',')]:
        rs = [U.per_backbone(base, pdb, a.n, 0.1, device, bias_aa=vec(b)) for (_, _, pdb) in bb]
        sn = float(np.mean([r['surface_net'] for r in rs]))
        rec = float(np.mean([r['recovery'] for r in rs]))
        if base_sn is None:
            base_sn = sn
        sh = sn - base_sn
        rows.append({'bias': b, 'surf_net': sn, 'shift': sh, 'pct_gap': -sh / gap['surface_net'] * 100, 'recovery': rec})
        print(f"{b:>6}{sn:>10.4f}{sh:>9.4f}{-sh / gap['surface_net'] * 100:>6.0f}%{rec:>10.4f}", flush=True)
    if a.save_csv:
        pd.DataFrame(rows).to_csv(a.save_csv, index=False)
        print(f'[saved] {a.save_csv}')
    print('\nRead vs FT_alk (test: surface_net shift ~-0.26 at recovery ~0.53): does the hand-bias need to sacrifice MORE recovery to reach the same surface_net? If so, the learned readout is better-calibrated.')
def seeds_sweep(a, device, bb, gap):
    """epoch_00 from several seeds, all on the SAME val backbones: is the 1-epoch steer a stable
    property or a lucky single-seed snapshot? Reports surface_net shift + recovery mean +/- SD."""
    (base_sn, base_rec) = eval_model(U.load_model(a.base, device), bb, a.n, 0.1, device)
    print(f'base: surface_net {base_sn:.4f} recovery {base_rec:.4f}\n')
    ckpts = [c for c in a.ckpts.split(',') if c]
    print(f"{'run':<28}{'surf_net':>10}{'shift':>9}{'%gap':>7}{'recovery':>10}")
    (shifts, recs) = ([], [])
    for c in ckpts:
        (sn, rec) = eval_model(U.load_model(c, device), bb, a.n, 0.1, device)
        sh = sn - base_sn
        shifts.append(sh)
        recs.append(rec)
        run = os.path.basename(os.path.dirname(os.path.dirname(c)))
        print(f"{run:<28}{sn:>10.4f}{sh:>9.4f}{-100 * sh / gap['surface_net']:>6.0f}%{rec:>10.4f}")
    (shifts, recs) = (np.array(shifts), np.array(recs))
    fg = -shifts / gap['surface_net']
    print(f'\nepoch_0 across {len(ckpts)} seeds: surface_net shift {shifts.mean():.4f} +/- {shifts.std():.4f} ({100 * fg.mean():.0f}% +/- {100 * fg.std():.0f}% of gap) | recovery {recs.mean():.4f} +/- {recs.std():.4f} (drop {100 * (base_rec - recs.mean()):.1f} pp)')
    print('small SD => the published epoch_0 is a stable property, not a single-seed fluke.')
def sweeps_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['checkpoint', 'temperature', 'seeds', 'logitbias'])
    ap.add_argument('--mpnn', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--ckpt_dir')
    ap.add_argument('--ckpt')
    ap.add_argument('--ckpts', default='', help='comma-separated epoch_00 checkpoints from different seeds (--mode seeds)')
    ap.add_argument('--save_csv', default='', help='write sweep rows to this CSV (for figures.py process curves)')
    ap.add_argument('--select', default='earliest', choices=['earliest', 'maxsteer'], help='earliest = first epoch clearing guardrail w/ >=min_gap (pre-reg); maxsteer = max %gap within guardrail')
    ap.add_argument('--select_out', default='', help='write the selected epoch number (NN) to this file')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--max_bb', type=int, default=30)
    ap.add_argument('--guardrail', type=float, default=0.03)
    ap.add_argument('--min_gap', type=float, default=0.25)
    ap.add_argument('--temps', default='0.1,0.5,1.0')
    ap.add_argument('--biases', default='0,0.5,1,2,3', help='logit-bias strengths (--mode logitbias)')
    ap.add_argument('--cohort', default='alkaline', choices=['alkaline', 'acid'])
    a = ap.parse_args()
    U.set_cohort(a.cohort)
    sys.path.insert(0, a.mpnn)
    device = U.pick_device()
    gap = U.natural_gap()
    bb = [b for b in U.collect_backbones(a.split) if b[1] == 'neutralophile'][:a.max_bb]
    print(f'{a.mode} sweep on {len(bb)} neutralophile {a.split} backbones x n={a.n}; device {device}')
    if a.mode == 'checkpoint':
        checkpoint_sweep(a, device, bb, gap)
    elif a.mode == 'seeds':
        seeds_sweep(a, device, bb, gap)
    elif a.mode == 'logitbias':
        logitbias_sweep(a, device, bb, gap)
    else:
        temperature_sweep(a, device, bb, gap)
def sweeps__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sweeps_main()

# ---------- from evaluate.py ----------
def evaluate_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mpnn', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--ft_alk', required=True)
    ap.add_argument('--ft_neu', default='')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--temp', type=float, default=0.1)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cohort', default='alkaline', choices=['alkaline', 'acid'])
    a = ap.parse_args()
    U.set_cohort(a.cohort)
    sys.path.insert(0, a.mpnn)
    device = U.pick_device()
    print('device:', device)
    models = {'base': U.load_model(a.base, device), 'FT_alk': U.load_model(a.ft_alk, device)}
    if a.ft_neu:
        models['FT_neu'] = U.load_model(a.ft_neu, device)
    bb = U.collect_backbones('test')
    if a.smoke:
        bb = bb[:3] + bb[-2:]
    print(f'evaluating {len(bb)} test backbones x {len(models)} models x {a.n} designs')
    rows = []
    for (acc, grp, pdb) in bb:
        for (mname, model) in models.items():
            with torch.no_grad():
                agg = U.per_backbone(model, pdb, a.n, a.temp, device)
            agg.update(acc=acc, group=grp, model=mname)
            rows.append(agg)
        print(f'  {acc} ({grp}) done', flush=True)
    df = pd.DataFrame(rows)
    outdir = U.OUTPUTS / 'evaluation'
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / ('axis_eval_designs_smoke.csv' if a.smoke else 'axis_eval_designs.csv')
    df.to_csv(csv, index=False)
    if a.smoke:
        print(f'[smoke] wrote {csv.name} ({df.acc.nunique()} backbones) - NOT the locked CSV/verdict')
        print(df.groupby('model')[U.AXIS + ['recovery']].mean())
        return
    verdict(df, outdir)
def verdict(df, outdir):
    gap = U.natural_gap()
    lines = ['LOCKED-criteria verdict (docs/EVALUATION_LOCKED.md)\n']
    piv = df[df.group == 'neutralophile'].pivot(index='acc', columns='model')
    pvals = {}
    for k in U.AXIS:
        d = (piv[k, 'FT_alk'] - piv[k, 'base']).dropna()
        try:
            (_, p) = wilcoxon(d, alternative='less' if U.DIRECTION[k] < 0 else 'greater')
        except Exception:
            p = np.nan
        frac = (np.sign(d) == U.DIRECTION[k]).mean()
        shift = d.mean()
        fgap = -shift / gap[k] if gap.get(k) else np.nan
        pvals[k] = (shift, fgap, frac, p)
    order = sorted(pvals, key=lambda k: pvals[k][3])
    mlen = len(order)
    holm = {k: min(1.0, pvals[k][3] * (mlen - i)) for (i, k) in enumerate(order)}
    lines.append(f"{'metric':<24}{'shift':>9}{'%gap':>8}{'%dir':>7}{'p_raw':>10}{'p_holm':>10}")
    for k in U.AXIS:
        (s, fg, fr, p) = pvals[k]
        lines.append(f'{k:<24}{s:9.4f}{100 * fg:7.0f}%{100 * fr:6.0f}%{p:10.2e}{holm[k]:10.2e}')
    (rb, rf) = (piv['recovery', 'base'], piv['recovery', 'FT_alk'])
    (nb, nf) = (piv['nll', 'base'], piv['nll', 'FT_alk'])
    drec = (rf - rb).mean()
    dnll = (nf - nb).mean()
    lines.append(f'\nrecovery base {rb.mean():.3f} -> FT_alk {rf.mean():.3f} (Δ {drec:+.3f} pp-frac)')
    lines.append(f'native NLL base {nb.mean():.3f} -> FT_alk {nf.mean():.3f} (Δ {dnll:+.3f} nats)')
    for k in U.OFF_AXIS:
        lines.append(f"off-axis {k}: Δ {(piv[k, 'FT_alk'] - piv[k, 'base']).dropna().mean():+.4f}")
    _shift = lambda k: (piv[k, 'FT_alk'] - piv[k, 'base']).dropna().mean()
    if ('core_net', 'FT_alk') in piv:
        lines.append(f"\nspecificity: surface_net Δ {_shift('surface_net'):+.4f} vs core_net Δ {_shift('core_net'):+.4f} | surface_acidic Δ {_shift('surface_acidic'):+.4f} vs core_acidic Δ {_shift('core_acidic'):+.4f} (surface-targeted if |surface| >> |core|)")
    if ('aa_entropy', 'FT_alk') in piv:
        lines.append(f"collapse: aa-entropy base {piv['aa_entropy', 'base'].mean():.2f} -> FT_alk {piv['aa_entropy', 'FT_alk'].mean():.2f} bits | diversity base {piv['diversity', 'base'].mean():.3f} -> FT_alk {piv['diversity', 'FT_alk'].mean():.3f}")
    if 'FT_neu' in df.model.unique():
        dn = (piv['surface_net', 'FT_neu'] - piv['surface_net', 'base']).dropna()
        lines.append(f"\nsymmetric FT_neu surface_net shift {dn.mean():+.4f} (vs FT_alk {pvals['surface_net'][0]:+.4f})")
    (s, fg, fr, p) = pvals['surface_net']
    primary = holm['surface_net'] < 0.05 and fg > 0
    magnitude = fg >= 0.25
    guard = drec >= -0.03 and dnll <= 0.1
    decision = 'PASS' if primary and magnitude and guard else 'PARTIAL' if primary and guard else 'FAIL'
    lines.append(f"\nPRIMARY(surface_net) p_holm {holm['surface_net']:.2e} dir {('ok' if fg > 0 else 'WRONG')} | magnitude {100 * fg:.0f}% of gap (>=25%? {magnitude}) | guardrail {('ok' if guard else 'FAIL')}")
    lines.append(f'\n==> DECISION: {decision}')
    txt = '\n'.join(lines)
    print(txt)
    (outdir / 'verdict.txt').write_text(txt)
def evaluate__entry():
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    evaluate_main()

# ---------- from run_proteinmpnn_surface.py ----------
run_proteinmpnn_surface_HERE = Path(__file__).resolve().parent
run_proteinmpnn_surface_ROOT = run_proteinmpnn_surface_HERE.parents[1]
MPNN = run_proteinmpnn_surface_ROOT / 'finetune' / 'third_party' / 'ProteinMPNN'
run_proteinmpnn_surface_FT = run_proteinmpnn_surface_ROOT / 'finetune' / 'outputs'
GEN = run_proteinmpnn_surface_ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'generation'
WORK = GEN / 'mpnn_work'
N_DESIGNS = 8
TEMP = '0.1'
WEIGHTS = {'base': (MPNN / 'vanilla_model_weights', 'v_48_020'), 'AlkSecMPNN': (run_proteinmpnn_surface_FT / 'alkaliphile_v1' / 'model_weights', 'epoch_best'), 'NeuSecMPNN': (run_proteinmpnn_surface_FT / 'neutralophile_control_v1' / 'model_weights', 'epoch_best')}
def run(cmd, **kw):
    print('  $', ' '.join((str(c) for c in cmd)))
    subprocess.run([str(c) for c in cmd], check=True, **kw)
def ensure_weights(display, folder, stem):
    """protein_mpnn_run.py expects a 'noise_level' key that the finetune trainer did
    not save; add it (0.2 A training backbone noise) to a patched copy if missing."""
    import torch
    src = Path(folder) / f'{stem}.pt'
    ck = torch.load(src, map_location='cpu', weights_only=False)
    if 'noise_level' in ck:
        return (str(folder) + '/', stem)
    ck['noise_level'] = 0.2
    wdir = WORK / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    torch.save(ck, wdir / f'{display}.pt')
    return (str(wdir) + '/', display)
def run_proteinmpnn_surface_main():
    targets = pd.read_csv(GEN / 'secreted_targets.csv')
    surf = json.load(open(GEN / 'secreted_surface_positions.json'))
    WORK.mkdir(parents=True, exist_ok=True)
    pdb_dir = WORK / 'pdbs'
    if pdb_dir.exists():
        shutil.rmtree(pdb_dir)
    pdb_dir.mkdir()
    for (_, r) in targets.iterrows():
        shutil.copy(r['structure_path'], pdb_dir / f"{r['name']}.pdb")
    parsed = WORK / 'parsed.jsonl'
    run([sys.executable, MPNN / 'helper_scripts' / 'parse_multiple_chains.py', '--input_path', pdb_dir, '--output_path', parsed], cwd=MPNN)
    fixed = {}
    for (_, r) in targets.iterrows():
        name = r['name']
        surf_set = set(surf[name]['surface'])
        core_1idx = [i + 1 for i in range(int(r['n_res'])) if i not in surf_set]
        fixed[name] = {'A': core_1idx}
    fixed_path = WORK / 'fixed_positions.jsonl'
    fixed_path.write_text(json.dumps(fixed))
    rows = []
    for (model, (folder0, stem0)) in WEIGHTS.items():
        (folder, stem) = ensure_weights(model, folder0, stem0)
        out = WORK / 'out' / model
        out.mkdir(parents=True, exist_ok=True)
        print(f'\n=== ProteinMPNN {model} ({stem}) ===')
        run([sys.executable, MPNN / 'protein_mpnn_run.py', '--jsonl_path', parsed, '--fixed_positions_jsonl', fixed_path, '--path_to_model_weights', folder, '--model_name', stem, '--num_seq_per_target', N_DESIGNS, '--sampling_temp', TEMP, '--out_folder', out, '--batch_size', 1], cwd=MPNN)
        for fa in (out / 'seqs').glob('*.fa'):
            name = fa.stem
            recs = _read_fasta(fa)
            for (k, (_, seq)) in enumerate(recs[1:]):
                rows.append(dict(name=name, model=model, sample_idx=k, sequence=seq))
    df = pd.DataFrame(rows)
    df.to_csv(GEN / 'proteinmpnn_designs.csv', index=False)
    print(f'\nwrote proteinmpnn_designs.csv ({len(df)} designs; {df.name.nunique()} targets x {df.model.nunique()} models)')
def _read_fasta(path):
    (recs, h, s) = ([], None, [])
    for line in open(path):
        line = line.rstrip()
        if line.startswith('>'):
            if h is not None:
                recs.append((h, ''.join(s)))
            (h, s) = (line[1:], [])
        elif line:
            s.append(line)
    if h is not None:
        recs.append((h, ''.join(s)))
    return recs
def run_proteinmpnn_surface__entry():
    run_proteinmpnn_surface_main()

# ---------- from ft020_base_vs_ft_self_consistency.py ----------
def ft020_base_vs_ft_self_consistency__entry():
    matplotlib.use('Agg')
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
    sys.path.insert(0, os.path.join(ROOT, 'design'))
    BASE_DIR = os.path.join(ROOT, 'design', 'arc_downloads', 'rank001_flat')
    OUT = os.path.join(HERE, 'outputs')
    FT_CSV = os.path.join(OUT, 'ft020_self_consistency_vs_afdb.csv')
    ref_path = dict(zip(dc.load_inputs().uniprot_id, dc.load_inputs().structure_path))

    def ca(path):
        chain = next(get_structure(path).get_chains())
        (coords, seq) = get_residue_data(chain)
        return (np.asarray(coords, float), seq)

    def mean_plddt(path):
        vals = [float(l[60:66]) for l in open(path) if l.startswith('ATOM') and l[12:16].strip() == 'CA']
        return float(np.mean(vals)) if vals else np.nan

    def p_label(p):
        if p < 0.001:
            return 'p<0.001'
        if p < 0.01:
            return f'p={p:.3f}'
        return f'p={p:.2f}'
    ref_ca = {u: ca(p) for (u, p) in ref_path.items()}
    rows = []
    for p in glob.glob(os.path.join(BASE_DIR, '*__ProteinMPNN__*rank_001*.pdb')):
        fid = os.path.basename(p).split('_unrelaxed_rank')[0]
        m = re.match('(.+?)__(.+?)__s(\\d+)$', fid)
        if not m or m.group(1) not in ref_ca:
            continue
        (uni, s) = (m.group(1), int(m.group(3)))
        (dc_, ds) = ca(p)
        (rc, rs) = ref_ca[uni]
        r = tm_align(dc_, rc, ds, rs)
        rows.append(dict(uniprot_id=uni, model='ProteinMPNN_v020(base)', sample_idx=s, scTM=r.tm_norm_chain2, scRMSD=r.rmsd, pLDDT=mean_plddt(p)))
    base = pd.DataFrame(rows)
    print(f'scored {len(base)} base ProteinMPNN(v_48_020) designs; templates={base.uniprot_id.nunique()}')
    ft = pd.read_csv(FT_CSV)
    comb = pd.concat([ft, base], ignore_index=True)
    comb.to_csv(os.path.join(OUT, 'ft020_self_consistency_vs_afdb_with_base.csv'), index=False)
    print('\n=== per-model self-consistency vs AFDB backbone (v_48_020) ===')
    print(comb.groupby('model')[['scTM', 'scRMSD', 'pLDDT']].agg(['mean', 'median']).round(3))
    b = base.groupby('uniprot_id')[['scTM', 'scRMSD', 'pLDDT']].mean()
    print('\n=== fine-tuned vs base (v_48_020), paired by template, Wilcoxon signed-rank ===')
    tests = {}
    for ftm in ['AlkSecMPNN_020', 'AcidSecMPNN_020']:
        f = comb[comb.model == ftm].groupby('uniprot_id')[['scTM', 'scRMSD', 'pLDDT']].mean()
        c = b.index.intersection(f.index)
        for met in ['scTM', 'scRMSD', 'pLDDT']:
            d = f.loc[c, met] - b.loc[c, met]
            try:
                pv = wilcoxon(f.loc[c, met], b.loc[c, met]).pvalue
            except ValueError:
                pv = float('nan')
            tests[ftm, met] = (d.mean(), pv)
            print(f'  {ftm:18s} {met:6s}  FT={f.loc[c, met].mean():.3f}  base={b.loc[c, met].mean():.3f}  delta={d.mean():+.3f}  p={pv:.3f}')
    plot = comb.groupby(['model', 'uniprot_id'], as_index=False)[['scTM', 'scRMSD', 'pLDDT']].mean()
    order = ['ProteinMPNN_v020(base)', 'AlkSecMPNN_020', 'AcidSecMPNN_020', 'WT_singleseq(control)']
    labels = ['ProteinMPNN\nbase', 'AlkSecMPNN', 'AcidSecMPNN', 'WT\nsingle-seq']
    colors = ['#7f8c8d', '#2c7fb8', '#d95f02', '#bdbdbd']
    metrics = [('scTM', 'Self-consistency TM', 'higher is better'), ('scRMSD', 'Cα-RMSD to input (Å)', 'lower is better'), ('pLDDT', 'Refold pLDDT', 'higher is better')]
    (fig, axes) = plt.subplots(1, 3, figsize=(12.8, 4.2))
    rng = np.random.default_rng(3)
    for (ax, (met, title, subtitle)) in zip(axes, metrics):
        vals = [plot.loc[plot.model == m, met].dropna().values for m in order]
        bp = ax.boxplot(vals, positions=np.arange(len(order)), widths=0.55, patch_artist=True, showfliers=False)
        for (patch, color) in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
        for part in ['whiskers', 'caps', 'medians']:
            for artist in bp[part]:
                artist.set_color('#333333')
        for (i, v) in enumerate(vals):
            jitter = rng.normal(0, 0.035, size=len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, s=14, color=colors[i], alpha=0.75, linewidths=0)
        ymax = max((max(v) for v in vals if len(v)))
        ymin = min((min(v) for v in vals if len(v)))
        pad = (ymax - ymin) * 0.18 if ymax > ymin else 1
        ax.set_ylim(ymin - pad * 0.25, ymax + pad)
        for (i, ftm) in [(1, 'AlkSecMPNN_020'), (2, 'AcidSecMPNN_020')]:
            (delta, p) = tests[ftm, met]
            ax.text(i, ymax + pad * 0.1, f'Δ={delta:+.2f}\n{p_label(p)}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_title(f'{title}\n{subtitle}', fontsize=10)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('v_48_020 fine-tuned design refolds vs AFDB input backbone', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ft020_self_consistency_vs_afdb_with_base.png'), dpi=150, bbox_inches='tight')
    print('\nSaved ft020_self_consistency_vs_afdb_with_base.csv')
    print('Saved ft020_self_consistency_vs_afdb_with_base.png')

# ---------- from ft020_self_consistency_vs_afdb.py ----------
def ft020_self_consistency_vs_afdb__entry():
    matplotlib.use('Agg')
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
    sys.path.insert(0, os.path.join(ROOT, 'design'))
    AF2_DIR = os.path.join(ROOT, 'design', 'outputs', 'colabfold_out_ft020')
    OUT = os.path.join(HERE, 'outputs')
    os.makedirs(OUT, exist_ok=True)
    ref_path = dict(zip(dc.load_inputs().uniprot_id, dc.load_inputs().structure_path))

    def ca(path):
        chain = next(get_structure(path).get_chains())
        (coords, seq) = get_residue_data(chain)
        return (np.asarray(coords, float), seq)

    def mean_plddt(path):
        vals = [float(l[60:66]) for l in open(path) if l.startswith('ATOM') and l[12:16].strip() == 'CA']
        return float(np.mean(vals)) if vals else np.nan

    def fold_id(path):
        return os.path.basename(path).split('_unrelaxed_rank')[0]
    pdbs = glob.glob(os.path.join(AF2_DIR, '*rank_001*.pdb'))
    print(f'{len(pdbs)} rank-1 structures in {AF2_DIR}')
    ref_ca = {u: ca(p) for (u, p) in ref_path.items()}
    rows = []
    for p in pdbs:
        fid = fold_id(p)
        if fid.endswith('__WT'):
            (uni, model, s) = (fid[:-4], 'WT_singleseq(control)', -1)
        else:
            m = re.match('(.+?)__(.+?)__s(\\d+)$', fid)
            if not m:
                continue
            (uni, model, s) = (m.group(1), m.group(2), int(m.group(3)))
        if uni not in ref_ca:
            continue
        (dcoords, dseq) = ca(p)
        (rcoords, rseq) = ref_ca[uni]
        r = tm_align(dcoords, rcoords, dseq, rseq)
        rows.append(dict(uniprot_id=uni, model=model, sample_idx=s, scTM=r.tm_norm_chain2, scRMSD=r.rmsd, pLDDT=mean_plddt(p)))
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(OUT, 'ft020_self_consistency_vs_afdb.csv'), index=False)
    print(f'\n{len(sc)} structures scored ->', os.path.join(OUT, 'ft020_self_consistency_vs_afdb.csv'))
    print('\n=== per-model self-consistency vs AFDB backbone ===')
    print(sc.groupby('model')[['scTM', 'scRMSD', 'pLDDT']].agg(['mean', 'median']).round(3))
    ctrl = sc[sc.model == 'WT_singleseq(control)'].set_index('uniprot_id')['scTM']
    print('\n=== FT designs vs WT single-seq control (paired by template, Wilcoxon) ===')
    for ft in ['AlkSecMPNN_020', 'AcidSecMPNN_020']:
        f = sc[sc.model == ft].groupby('uniprot_id')['scTM'].mean()
        c = f.index.intersection(ctrl.index)
        try:
            pv = wilcoxon(f.loc[c], ctrl.loc[c]).pvalue
        except ValueError:
            pv = float('nan')
        print(f'  {ft:20s} scTM={f.mean():.3f}  control={ctrl.loc[c].mean():.3f}  delta={(f.loc[c] - ctrl.loc[c]).mean():+.3f}  p={pv:.2e}')
    (fig, ax) = plt.subplots(1, 3, figsize=(13, 4))
    order = ['AlkSecMPNN_020', 'AcidSecMPNN_020', 'WT_singleseq(control)']
    sc['model'] = pd.Categorical(sc['model'], order, ordered=True)
    for (a, met, lab) in zip(ax, ['scTM', 'scRMSD', 'pLDDT'], ['self-consistency TM (vs AFDB)', 'self-consistency RMSD (A)', 'design pLDDT']):
        sc.boxplot(column=met, by='model', ax=a, grid=False)
        a.set_title(lab)
        a.set_xlabel('')
        a.set_xticklabels([t.get_text().replace('_singleseq(control)', '').replace('MPNN_020', 'MPNN') for t in a.get_xticklabels()], rotation=15)
    plt.suptitle('v_48_020 fine-tuned design self-consistency vs AFDB backbone (25 templates x 8 designs)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'ft020_self_consistency_vs_afdb.png'), dpi=150, bbox_inches='tight')
    print('\nSaved ft020_self_consistency_vs_afdb.png')

_STEPS = {
    'train': train__entry,
    'sweeps': sweeps__entry,
    'evaluate': evaluate__entry,
    'run-proteinmpnn-surface': run_proteinmpnn_surface__entry,
    'ft020-base-vs-ft-self-consistency': ft020_base_vs_ft_self_consistency__entry,
    'ft020-self-consistency-vs-afdb': ft020_self_consistency_vs_afdb__entry,
}

def main(argv=None):
    import sys
    argv = sys.argv if argv is None else argv
    if len(argv) < 2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    sys.argv = [argv[0]] + argv[2:]
    _STEPS[argv[1]](); return 0

if __name__ == '__main__':
    raise SystemExit(main())

