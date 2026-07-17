"""Tests for esm_generation pure logic (no ESM/GPU needed).
Run: python test_esm_generation.py   (or via pytest)
"""
import numpy as np

from decoding_bias.finetune.esm_generation import (
    CANONICAL_AA, aa_probs_from_logits, acidic_propensity,
    sample_from_probs, iterative_infill)


def test_aa_probs_from_logits_selects_and_renormalizes():
    # vocab of 5 tokens; "amino-acid" tokens are ids 1 and 3
    logits = np.array([0.0, np.log(2), 0.0, np.log(2), 0.0])
    p = aa_probs_from_logits(logits, aa_token_ids=[1, 3])
    assert p.shape == (2,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.allclose(p, [0.5, 0.5])


def test_acidic_propensity_known_values():
    aa = CANONICAL_AA
    di, ei = aa.index("D"), aa.index("E")
    ki, ri = aa.index("K"), aa.index("R")
    L = 3
    m = np.zeros((L, 20))
    m[0, di] = 1.0          # all acidic  -> +1
    m[1, ki] = 1.0          # all basic   -> -1
    m[2, :] = 1.0 / 20      # uniform      -> 0
    prop = acidic_propensity(m, aa)
    assert prop.shape == (3,)
    assert abs(prop[0] - 1.0) < 1e-9
    assert abs(prop[1] + 1.0) < 1e-9
    assert abs(prop[2]) < 1e-9


def test_sample_from_probs_argmax_at_low_temperature():
    probs = np.array([0.1, 0.7, 0.2] + [0.0] * 17)
    rng = np.random.default_rng(0)
    idx = sample_from_probs(probs, temperature=1e-6, rng=rng)
    assert idx == 1


def test_sample_from_probs_is_reproducible():
    probs = np.full(20, 1 / 20)
    draws1 = [sample_from_probs(probs, 1.0, np.random.default_rng(7)) for _ in range(1)]
    draws2 = [sample_from_probs(probs, 1.0, np.random.default_rng(7)) for _ in range(1)]
    assert draws1 == draws2


def test_iterative_infill_respects_fixed_positions_and_length():
    # mock model that always wants "D" everywhere
    di = CANONICAL_AA.index("D")
    def predict_fn(seq, pos):
        p = np.zeros(20); p[di] = 1.0
        return p
    wt = "KKKKKK"
    design = iterative_infill(wt, predict_fn, n_passes=1, temperature=0.0,
                              seed=0, fixed_positions=[0, 5])
    assert len(design) == len(wt)
    assert design[0] == "K" and design[5] == "K"      # fixed kept
    assert design[1:5] == "DDDD"                        # rest redesigned to D


def test_iterative_infill_reproducible():
    rng_probs = np.random.default_rng(1).random(20)
    rng_probs /= rng_probs.sum()
    def predict_fn(seq, pos):
        return rng_probs
    wt = "ACDEFGHIKL"
    a = iterative_infill(wt, predict_fn, n_passes=2, temperature=0.5, seed=3)
    b = iterative_infill(wt, predict_fn, n_passes=2, temperature=0.5, seed=3)
    assert a == b and len(a) == len(wt)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")
