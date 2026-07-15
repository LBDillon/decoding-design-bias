#!/usr/bin/env python3
"""Workflow-integrity smoke tests: package imports, CLI wiring, scientific invariants.

Run directly (`python tests/test_imports.py`) or via pytest.
"""
import importlib


def test_core_packages():
    for pkg in ["pandas", "numpy", "scipy", "statsmodels", "sklearn",
                "matplotlib", "Bio", "tqdm", "yaml"]:
        importlib.import_module(pkg)


def test_package_modules():
    for mod in [
        "decoding_bias", "decoding_bias.catalog", "decoding_bias.config",
        "decoding_bias.data", "decoding_bias.cli", "decoding_bias.manifest",
        "decoding_bias.verify", "decoding_bias.plotting",
        "decoding_bias.analysis.variance", "decoding_bias.analysis.importance",
        "decoding_bias.analysis.elo", "decoding_bias.analysis.elo_rating",
        "decoding_bias.analysis.pca",
        "decoding_bias.features.sequence_features",
        "decoding_bias.features.structural_features",
    ]:
        importlib.import_module(mod)


def test_cli_parser():
    from decoding_bias.cli import build_parser
    p = build_parser()
    # every scientific stage is a subcommand
    subs = p._subparsers._group_actions[0].choices
    for stage in ["dataset", "variance", "taxonomy", "pca", "importance",
                  "design", "finetune", "pdb-cohort", "score", "figures", "verify"]:
        assert stage in subs, f"missing subcommand: {stage}"


def test_scientific_invariants():
    from decoding_bias import catalog
    assert len(catalog.BIOPHYS_14) == 14
    assert len(catalog.SEQUENCE_FEATURES_14) == 9
    assert len(catalog.STRUCTURE_FEATURES_14) == 5
    assert len(catalog.FULL_COHORT) == 14
    assert len(catalog.IMPORTANCE_PANEL) == 15
    assert len(catalog.FINETUNE_020) == 3
    assert catalog.DOMAINS == ["Archaea", "Bacteria", "Eukaryota"]


def test_config_and_data():
    from decoding_bias.config import Config
    cfg = Config.load()
    assert cfg.analysis_table.exists(), f"analysis table missing: {cfg.analysis_table}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll workflow-integrity checks passed.")
