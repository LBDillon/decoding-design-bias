#!/usr/bin/env python3
"""Scientific-equivalence test: reproducible stages match tests/reference/.

Runs the fast reproducible stages (variance, importance, PCA, dataset) and diffs
against the committed reference artifacts. This is the regression guard for the
consolidation. Elo (slow) is covered by `decoding-bias verify --full`.

Run directly or via pytest.
"""
from decoding_bias.config import Config
from decoding_bias import verify


def test_reproducible_stages_match_reference():
    cfg = Config.load()
    ok = verify.run(cfg, fast=True)
    assert ok, "a reproducible stage diverged from tests/reference/ - see output above"


if __name__ == "__main__":
    test_reproducible_stages_match_reference()
    print("\nEquivalence test passed.")
