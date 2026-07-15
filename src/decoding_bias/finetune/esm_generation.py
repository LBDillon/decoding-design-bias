"""ESM2 generation / per-position probability tools for the malleability comparison
(apples-to-apples with ProteinMPNN designs, Reviewer R1.4).

Two layers:
  * pure logic (numpy only, unit-tested with a mock model): aa_probs_from_logits,
    acidic_propensity, sample_from_probs, iterative_infill.
  * ESM wiring (torch/transformers, imported lazily; exercised on GPU/Colab):
    esm_position_distributions, make_esm_predict_fn.

Phase 1 uses esm_position_distributions to get an (L, 20) probability map per
protein (base vs fine-tuned) for the substitution heatmap. Phase 2 uses
iterative_infill (Gibbs-style masked in-filling) to actually generate sequences.
"""
import numpy as np

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"   # 20, fixed order
ACIDIC = ("D", "E")
BASIC = ("K", "R")


# ---- pure logic -----------------------------------------------------------
def aa_probs_from_logits(logits, aa_token_ids):
    """Softmax over the full vocab, select the amino-acid tokens, renormalise
    to a distribution over just the amino acids."""
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max()
    p = np.exp(z)
    p /= p.sum()
    p_aa = p[np.asarray(aa_token_ids, dtype=int)]
    s = p_aa.sum()
    return p_aa / s if s > 0 else p_aa


def acidic_propensity(prob_matrix, aa_order=CANONICAL_AA, acidic=ACIDIC, basic=BASIC):
    """Per-position P(acidic) - P(basic) over an (L, 20) probability matrix."""
    m = np.asarray(prob_matrix, dtype=float)
    ai = [aa_order.index(a) for a in acidic]
    bi = [aa_order.index(a) for a in basic]
    return m[:, ai].sum(axis=1) - m[:, bi].sum(axis=1)


def sample_from_probs(probs, temperature, rng):
    """Sample an index from a (20,) distribution with temperature. temperature<=0
    is greedy (argmax)."""
    probs = np.asarray(probs, dtype=float)
    if temperature <= 0:
        return int(np.argmax(probs))
    # temperature in log-space with max-subtraction, so tiny temperatures reduce
    # to argmax rather than underflowing to a uniform distribution.
    with np.errstate(divide="ignore"):
        logits = np.log(probs) / temperature
    finite = logits[np.isfinite(logits)]
    if finite.size == 0:
        return int(rng.integers(len(probs)))
    p = np.exp(logits - finite.max())
    tot = p.sum()
    p = p / tot if tot > 0 else np.full_like(probs, 1.0 / len(probs))
    return int(rng.choice(len(p), p=p))


def iterative_infill(seq, predict_fn, aa_order=CANONICAL_AA, n_passes=1,
                     temperature=0.1, seed=0, fixed_positions=None):
    """Gibbs-style masked in-filling design. Visits positions in random order,
    redesigns each from ``predict_fn(current_sequence, position) -> (20,) probs``,
    for ``n_passes`` passes. ``fixed_positions`` are held at their input residue.
    """
    rng = np.random.default_rng(seed)
    s = list(seq)
    fixed = set(fixed_positions or [])
    for _ in range(n_passes):
        for pos in rng.permutation(len(s)):
            pos = int(pos)
            if pos in fixed:
                continue
            probs = predict_fn("".join(s), pos)
            s[pos] = aa_order[sample_from_probs(probs, temperature, rng)]
    return "".join(s)


# ---- ESM wiring (torch/transformers; run on GPU/Colab) --------------------
def _aa_token_ids(tokenizer, aa_order=CANONICAL_AA):
    return [tokenizer.convert_tokens_to_ids(a) for a in aa_order]


def esm_position_distributions(seq, tokenizer, model, device, batch=64,
                               aa_order=CANONICAL_AA):
    """(L, 20) amino-acid probability map: for each residue position, mask it and
    read the model's renormalised distribution over the 20 amino acids."""
    import torch

    aa_ids = _aa_token_ids(tokenizer, aa_order)
    enc = tokenizer(seq, return_tensors="pt", return_special_tokens_mask=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    special = enc["special_tokens_mask"].to(device)
    res_pos = (special[0] == 0).nonzero(as_tuple=False).flatten()
    if res_pos.numel() != len(seq):
        raise ValueError(f"{res_pos.numel()} residue tokens for {len(seq)} residues.")
    out = np.zeros((len(seq), len(aa_order)), dtype=float)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(seq), batch):
            end = min(start + batch, len(seq))
            chunk = res_pos[start:end]
            b = input_ids.repeat(chunk.numel(), 1)
            a = attn.repeat(chunk.numel(), 1)
            row = torch.arange(chunk.numel(), device=device)
            b[row, chunk] = tokenizer.mask_token_id
            logits = model(input_ids=b, attention_mask=a).logits[row, chunk, :]
            logits = logits.to(torch.float32).cpu().numpy()
            for j in range(end - start):
                out[start + j] = aa_probs_from_logits(logits[j], aa_ids)
    return out


def make_esm_predict_fn(tokenizer, model, device, aa_order=CANONICAL_AA):
    """Return predict_fn(current_seq, position) -> (20,) probs for iterative_infill:
    masks the single position in the current sequence and reads its AA distribution."""
    import torch

    aa_ids = _aa_token_ids(tokenizer, aa_order)

    def predict_fn(current_seq, position):
        enc = tokenizer(current_seq, return_tensors="pt", return_special_tokens_mask=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        special = enc["special_tokens_mask"].to(device)
        res_pos = (special[0] == 0).nonzero(as_tuple=False).flatten()
        tok = res_pos[position]
        input_ids[0, tok] = tokenizer.mask_token_id
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits[0, tok, :]
        return aa_probs_from_logits(logits.to(torch.float32).cpu().numpy(), aa_ids)

    return predict_fn
