"""Shared cosine-to-axis metric for the pH-feature malleability test.

A model's *score-preference direction* is the gradient of its (standardised) score
regressed on the acid-base feature-PCA axes (PC1, PC2), with +PC1 oriented acidic.
The malleability readout is the cosine of that gradient to the +PC1 (acidic) axis:
+1 = prefers acidic, -1 = prefers basic, 0 = off-axis. This is the same test the
paper applies to the structure models (AlkSecMPNN etc.); routing the ESM sequence-
model scores through the identical function keeps both model classes on one axis.

`bootstrap_cosine_ci` adds the resampling interval the paper's R1.4 flag requires.
"""
import numpy as np


def _std(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd else x - x.mean()


def acidbase_axis(features, acidic_index):
    """PCA on the charge/pH features -> the shared acid-base axis.

    Returns (pc1, pc2, loadings, varexp): pc1/pc2 are the standardised PC scores
    (pc1 oriented so + = acidic via the acidic feature at ``acidic_index``; pc2
    sign pinned the same way so it is deterministic), loadings is the 2xF PC-space
    loading matrix, varexp the fraction of variance for PC1/PC2. Same construction
    the structure-model cosine map uses, so every model lands on one axis."""
    x = np.asarray(features, dtype=float)
    z = (x - x.mean(0)) / x.std(0)
    zc = z - z.mean(0)
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    if vt[0, acidic_index] < 0:   # PC1: + = acidic
        vt[0] *= -1
    if vt[1, acidic_index] < 0:   # pin PC2 sign (cosine-to-PC1 is invariant to it)
        vt[1] *= -1
    scores = zc @ vt[:2].T
    pc1 = _std(scores[:, 0])
    pc2 = _std(scores[:, 1])
    varexp = (s ** 2 / (s ** 2).sum())[:2]
    return pc1, pc2, vt[:2].copy(), varexp


def preference_gradient(pc1, pc2, score):
    """Standardised-score regression gradient on (PC1, PC2): the 2-vector
    score-preference direction in the acid-base feature-PCA plane.

    ``pc1``/``pc2`` are the *shared* axis coordinates and must already be
    standardised over the full dataset (the axis is one fixed property of the
    dataset, not re-derived per model); only ``score`` is standardised here, over
    whatever rows the caller passes. This keeps every model on one common axis."""
    pc1 = np.asarray(pc1, dtype=float)
    pc2 = np.asarray(pc2, dtype=float)
    y = _std(score)
    X = np.column_stack([np.ones(len(y)), pc1, pc2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.array([beta[1], beta[2]])


def cosine_to_axis(pc1, pc2, score):
    """Cosine of the score-preference gradient to the +PC1 (acidic) axis."""
    g = preference_gradient(pc1, pc2, score)
    norm = np.linalg.norm(g)
    return float(g[0] / norm) if norm else float("nan")


def bootstrap_cosine_ci(pc1, pc2, score, n_boot=1000, seed=0, ci=0.95):
    """Return (point_cosine, ci_lo, ci_hi) via case-resampling bootstrap."""
    pc1 = np.asarray(pc1, dtype=float)
    pc2 = np.asarray(pc2, dtype=float)
    score = np.asarray(score, dtype=float)
    point = cosine_to_axis(pc1, pc2, score)
    rng = np.random.default_rng(seed)
    n = len(score)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = cosine_to_axis(pc1[idx], pc2[idx], score[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return point, float(lo), float(hi)
