"""
Fine-tune ProteinMPNN on the alkaliphile/neutralophile secretome jsonl (the same backbone
records used everywhere else: design/outputs/alkaliphile_parsed_{train,val}.jsonl, filtered by
role). Trains the INFERENCE ProteinMPNN class (protein_mpnn_utils) so the weights load directly
in evaluate_designs.py; featurizes with training.model_utils.featurize (avoids the tied_featurize
variable-length batch bug). Cluster-disjoint splits come from Stage D (encoded in the jsonl).

Usage:
  python train_finetune.py --mpnn finetune/third_party/ProteinMPNN --run alkaliphile_v1 \
       --role case   [--epochs 30 --lr 1e-4 --batch_tokens 3000]
  python train_finetune.py ... --run neutralophile_control_v1 --role control
"""
import sys, os, argparse, json, time, random, warnings
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = Path(__file__).resolve().parent; FT = HERE.parent; ROOT = FT.parent
JSONL = FT / "data" if (FT / "data").exists() else ROOT / "design" / "outputs"  # self-contained


def load_records(jsonl_dir, split, role, exclude_clade="", label="alkaliphile"):
    recs = []
    for line in open(Path(jsonl_dir) / f"{label}_parsed_{split}.jsonl"):
        r = json.loads(line)
        if role and r.get("role") != role: continue          # role="" => any role (negative control)
        if exclude_clade and r.get("clade") == exclude_clade: continue
        r["masked_list"] = ["A"]; r["visible_list"] = []   # design chain A
        recs.append(r)
    return recs


def batches(recs, batch_tokens):
    recs = sorted(recs, key=lambda r: len(r["seq"]))
    out, cur, clen = [], [], 0
    for r in recs:
        L = len(r["seq"])
        if cur and (clen + L > batch_tokens or len(cur) >= 8):
            out.append(cur); cur, clen = [], 0
        cur.append(r); clen += L
    if cur: out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpnn", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--role", required=True, choices=["case", "control"])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_tokens", type=int, default=3000)
    ap.add_argument("--backbone_noise", type=float, default=0.2); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--jsonl_dir", default=str(JSONL), help="dir with alkaliphile_parsed_*.jsonl")
    ap.add_argument("--out_root", default=str(FT / "outputs"), help="output root dir")
    ap.add_argument("--save_every", type=int, default=0, help="also save epoch_NN.pt every N epochs")
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="linear LR warmup over N gradient steps (0 = constant lr, the locked default). "
                         "Use to test whether the 1-epoch steering is a no-warmup artefact.")
    ap.add_argument("--freeze", default="none", choices=["none", "wout"],
                    help="wout = freeze all but the output projection W_out (readout-only probe, steelman #2)")
    ap.add_argument("--schedule", default="const", choices=["const", "noam"],
                    help="const = Adam at --lr (locked recipe); noam = get_std_opt warmup "
                         "(original ProteinMPNN schedule, comparable to the acidophilic baseline). "
                         "noam ignores --lr/--warmup_steps and ramps lr gently.")
    ap.add_argument("--cohort", default="alkaline", choices=["alkaline", "acid"],
                    help="which cohort's *_parsed_*.jsonl to train on (alkaline=alkaliphile, acid=acidophile)")
    ap.add_argument("--exclude_clade", default="", help="drop this clade from training cases (leave-one-clade-out)")
    ap.add_argument("--frac", type=float, default=1.0, help="use this fraction of training cases (dose-response)")
    a = ap.parse_args()
    sys.path.insert(0, a.mpnn); sys.path.insert(0, str(Path(a.mpnn) / "training"))
    from model_utils import featurize, get_std_opt
    from protein_mpnn_utils import ProteinMPNN, loss_smoothed
    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    label = {"alkaline": "alkaliphile", "acid": "acidophile"}[a.cohort]
    tr = load_records(a.jsonl_dir, "train", a.role, a.exclude_clade, label)
    va = load_records(a.jsonl_dir, "val", a.role, a.exclude_clade, label)
    if a.exclude_clade: print(f"[exclude_clade] dropped {a.exclude_clade}; {len(tr)} train cases remain")
    if a.frac < 1.0:
        random.shuffle(tr); tr = tr[:max(1, int(len(tr) * a.frac))]
        print(f"[frac] using {a.frac:.0%} of train = {len(tr)} cases")
    if a.smoke: tr = tr[:a.smoke]; va = va[:max(2, a.smoke // 4)]
    print(f"[{a.run}] train {len(tr)} | val {len(va)} ({a.role}); device {device}")
    tr_b, va_b = batches(tr, a.batch_tokens), batches(va, a.batch_tokens)

    ckpt = torch.load(Path(a.mpnn) / "vanilla_model_weights" / "v_48_002.pt", map_location=device)
    model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=ckpt["num_edges"],
                        augment_eps=a.backbone_noise, dropout=a.dropout)
    model.to(device); model.load_state_dict(ckpt["model_state_dict"])
    if a.freeze == "wout":   # readout-only probe (steelman #2): only the output projection can move
        for nm, p in model.named_parameters(): p.requires_grad = nm.startswith("W_out")
        print(f"[freeze] training W_out only: {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
    train_params = [p for p in model.parameters() if p.requires_grad]
    if a.schedule == "noam":   # original ProteinMPNN warmup; comparable to the acidophilic baseline
        opt = get_std_opt(train_params, 128, 0)
        print("[opt] Noam warmup (get_std_opt: d_model=128, factor=2, warmup=4000) - lr ramps gently")
    else:
        opt = torch.optim.Adam(train_params, lr=a.lr, betas=(0.9, 0.98), eps=1e-9)
    gstep = 0  # global training-step counter, for optional LR warmup

    def lr_for(step):  # linear warmup to a.lr over warmup_steps, then constant (0 => always a.lr = locked behaviour)
        return a.lr * (step + 1) / a.warmup_steps if (a.warmup_steps and step < a.warmup_steps) else a.lr

    def epoch(bs, train):
        nonlocal gstep
        model.train(train); tot = acc = res = 0.0
        if train: random.shuffle(bs)
        for batch in bs:
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_enc = featurize(batch, device)
            mfl = mask * chain_M
            randn = torch.randn(chain_M.shape, device=device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_enc, randn)
                _, loss_av = loss_smoothed(S, log_probs, mfl)
                if train:
                    loss_av.backward(); torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                    if a.schedule == "const":
                        for g in opt.param_groups: g["lr"] = lr_for(gstep)
                    opt.step(); gstep += 1   # noam sets its own lr inside .step()
            nll = -log_probs.gather(-1, S.unsqueeze(-1)).squeeze(-1)   # [B,L]
            tf = (log_probs.argmax(-1) == S).float()
            w = mfl.sum().item(); tot += (nll * mfl).sum().item(); acc += (tf * mfl).sum().item(); res += w
        return tot / res, acc / res

    od = Path(a.out_root) / a.run; (od / "model_weights").mkdir(parents=True, exist_ok=True)
    (od / "log.txt").write_text(""); json.dump(vars(a), open(od / "config.json", "w"), indent=2)
    best = 1e9
    for ep in range(a.epochs):
        t0 = time.time(); trn, tra = epoch(tr_b, True)
        with torch.no_grad(): vn, vaa = epoch(va_b, False)
        line = f"epoch {ep} train_nll {trn:.3f} train_rec {tra:.3f} | val_nll {vn:.3f} val_rec {vaa:.3f} | lr {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s"
        print(line, flush=True); open(od / "log.txt", "a").write(line + "\n")
        save = {"model_state_dict": model.state_dict(), "num_edges": ckpt["num_edges"], "epoch": ep, "step": 0}
        torch.save(save, od / "model_weights" / "epoch_last.pt")
        if vn < best: best = vn; torch.save(save, od / "model_weights" / "epoch_best.pt")
        if a.save_every and (ep % a.save_every == 0 or ep == a.epochs - 1):
            torch.save(save, od / "model_weights" / f"epoch_{ep:02d}.pt")   # tradeoff-curve checkpoints
    print(f"[{a.run}] done. best val_nll {best:.3f}")


if __name__ == "__main__":
    main()
