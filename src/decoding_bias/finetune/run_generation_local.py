#!/usr/bin/env python
"""Local (CPU/MPS) version of the ESM2-35M generation pipeline - for when Colab
compute runs out. Self-contained: needs only torch + transformers (a plain MLM
fine-tune loop, no HF Trainer / datasets / accelerate).

By default it computes only what's MISSING from the per-template .npz maps in the
generation dir, so if you already have the base + AlkSecESM35M maps from Colab it
just trains the NeuSecESM35M control and merges it in. Use --force to redo all.

  # in an env with torch + transformers (e.g. esm2_local after
  #   conda run -n esm2_local python -m pip install "transformers>=4.46"):
  python run_generation_local.py                 # fill in missing models
  python run_generation_local.py --force         # recompute base + both FT
  python run_generation_local.py --phase2        # also generate sequences

Then make the figures (base env has biotite):
  python esm_design_heatmaps.py
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from esm_generation import (esm_position_distributions, make_esm_predict_fn,
                            iterative_infill, CANONICAL_AA)

BASE_MODEL = "facebook/esm2_t12_35M_UR50D"
DATA = ROOT / "outputs" / "esm35m_continual_pretraining" / "data"
GEN_DIR = ROOT / "outputs" / "esm35m_continual_pretraining" / "generation"
RUNS = HERE / "runs_local"
TEMPLATE_IDS = ["Q4R312", "P0A7R6", "A9A498", "A0A1D8PCG7", "Q9Z9J6"]
# model -> training-CSV stem (None = the frozen base model)
ARMS = {"base": None, "AlkSecESM35M": "alkaline_case", "NeuSecESM35M": "alkaline_neu"}


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_seqs(stem, split):
    return pd.read_csv(DATA / f"{stem}_{split}.csv")["sequence"].dropna().tolist()


def finetune_mlm(train_seqs, out_dir, epochs, lr, device, mlm_prob=0.15,
                 batch=4, max_len=1022, seed=1):
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
        tot, n = 0.0, 0
        for i in range(0, len(seqs), batch):
            enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len, return_special_tokens_mask=True)
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            special = enc["special_tokens_mask"].bool().to(device)
            labels = ids.clone()
            probs = torch.full(ids.shape, mlm_prob, device=device)
            probs[special] = 0.0
            masked = torch.bernoulli(probs).bool()
            labels[~masked] = -100
            # 80% -> [MASK], 10% -> random, 10% -> unchanged
            r = torch.rand(ids.shape, device=device)
            mask_tok = masked & (r < 0.8)
            rand_tok = masked & (r >= 0.8) & (r < 0.9)
            ids[mask_tok] = tok.mask_token_id
            ids[rand_tok] = torch.randint(vocab, ids.shape, device=device)[rand_tok]
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            tot += loss.item(); n += 1
        print(f"    epoch {ep + 1}/{epochs}  train_loss {tot / max(n, 1):.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return out_dir


def model_dir(name, epochs, lr, device):
    if ARMS[name] is None:
        return BASE_MODEL
    out = RUNS / f"{name}_e{epochs}"
    if (out / "config.json").exists():
        print(f"  reuse trained {name} at {out}")
        return str(out)
    print(f"  fine-tuning {name} ({epochs} ep) ...")
    tr = load_seqs(ARMS[name], "train")
    finetune_mlm(tr, out, epochs, lr, device)
    return str(out)


def templates(path=""):
    if path:
        d = pd.read_csv(path)
        namecol = "name" if "name" in d.columns else "uniprot_id"
        seqcol = "seq" if "seq" in d.columns else ("sequence" if "sequence" in d.columns else "wt_sequence")
        return list(zip(d[namecol], d[seqcol]))
    d = pd.read_csv(ROOT / "design" / "design_input_proteins.csv")
    d = d[d.uniprot_id.isin(TEMPLATE_IDS)].set_index("uniprot_id").loc[TEMPLATE_IDS]
    return list(zip(d.index, d.wt_sequence))


def existing(uid):
    f = GEN_DIR / f"{uid}_probs.npz"
    if not f.exists():
        return {}
    with np.load(f, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--force", action="store_true", help="recompute all models")
    ap.add_argument("--phase2", action="store_true", help="also generate sequences")
    ap.add_argument("--n_designs", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--redesign", choices=["surface", "all"], default="surface",
                    help="surface = redesign only surface positions (WT core fixed; "
                         "avoids the low-complexity collapse of full redesign).")
    ap.add_argument("--n_passes", type=int, default=2)
    ap.add_argument("--targets", default="", help="CSV with columns name/seq to design on "
                    "(default: the 5 built-in templates).")
    ap.add_argument("--surface_json", default="", help="surface-positions JSON for --targets "
                    "(default: generation/surface_positions.json).")
    ap.add_argument("--skip_maps", action="store_true", help="skip Phase-1 probability maps "
                    "(use for a design-only run on new targets).")
    ap.add_argument("--out_tag", default="", help="tag inserted into the designs filename.")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")
    tmpl = templates(args.targets)

    # which models are missing from at least one template map?
    if args.skip_maps:
        need = []
        print("skipping Phase-1 maps")
    else:
        have = {uid: set(existing(uid)) for uid, _ in tmpl}
        need = [m for m in ARMS if args.force or any(m not in have[uid] for uid, _ in tmpl)]
        print(f"models to compute: {need or 'none (all present)'}")

    # Phase 1: per-position maps for the missing models
    for name in need:
        path = model_dir(name, args.epochs, args.lr, device)
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()
        for uid, seq in tmpl:
            mat = esm_position_distributions(seq, tok, mdl, device)
            store = existing(uid)
            store.update({"seq": seq, "aa_order": np.array(list(CANONICAL_AA)), name: mat})
            np.savez(GEN_DIR / f"{uid}_probs.npz", **store)
            print(f"  {uid} {name}: {mat.shape}")
        del mdl

    # Phase 2 (optional): generate sequences with every model (base + both FT).
    # Surface-targeted redesign keeps the WT core fixed so ESM in-filling does not
    # collapse to low-complexity acidic runs (full-redesign failure mode).
    design_rows = []
    if args.phase2:
        import json
        fixed_by_uid = {}
        if args.redesign == "surface":
            surf_path = args.surface_json or (GEN_DIR / "surface_positions.json")
            surf = json.load(open(surf_path))
            for uid, seq in tmpl:
                surf_set = set(surf[uid]["surface"])
                fixed_by_uid[uid] = [i for i in range(len(seq)) if i not in surf_set]
        for name in ARMS:
            path = model_dir(name, args.epochs, args.lr, device)
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()
            pf = make_esm_predict_fn(tok, mdl, device)
            for uid, seq in tmpl:
                fixed = fixed_by_uid.get(uid)
                for k in range(args.n_designs):
                    des = iterative_infill(seq, pf, n_passes=args.n_passes,
                                           temperature=args.temperature, seed=k,
                                           fixed_positions=fixed)
                    design_rows.append(dict(uniprot_id=uid, model=name, sample_idx=k,
                                            redesign=args.redesign, sequence=des))
            print(f"  generated {args.n_designs}/template with {name}")
            del mdl

    if design_rows:
        tag = f"{args.out_tag}_" if args.out_tag else ""
        out = GEN_DIR / f"esm_designs_local_{tag}{args.redesign}.csv"
        df = pd.DataFrame(design_rows)
        if out.exists():
            df = pd.concat([pd.read_csv(out), df], ignore_index=True).drop_duplicates(
                ["uniprot_id", "model", "sample_idx"], keep="last")
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} designs)")
    print("done. Now run: python esm_design_heatmaps.py")


if __name__ == "__main__":
    main()
