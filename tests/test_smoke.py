"""Fast package, data-contract, and scientific-invariant checks."""
from __future__ import annotations

import importlib

import pandas as pd


def test_core_modules_import() -> None:
    for module in [
        "decoding_bias.catalog",
        "decoding_bias.config",
        "decoding_bias.data",
        "decoding_bias.reproduce",
        "decoding_bias.analysis.variance",
        "decoding_bias.analysis.elo",
        "decoding_bias.analysis.pca",
        "decoding_bias.analysis.importance",
        "decoding_bias.analysis.design",
        "decoding_bias.analysis.finetune",
    ]:
        importlib.import_module(module)


def test_cli_is_reviewer_scoped() -> None:
    from decoding_bias.cli import build_parser

    commands = build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == {
        "reproduce", "dataset", "variance", "taxonomy", "pca", "gam",
        "importance", "design", "finetune",
    }


def test_scientific_invariants_and_inputs() -> None:
    from decoding_bias import catalog
    from decoding_bias.config import Config
    from decoding_bias.reproduce import missing_inputs

    cfg = Config.load()
    assert len(catalog.BIOPHYS_14) == 14
    assert len(catalog.FULL_COHORT) == 14
    assert len(catalog.IMPORTANCE_PANEL) == 15
    assert catalog.DOMAINS == ["Archaea", "Bacteria", "Eukaryota"]
    assert not missing_inputs(cfg)

    main = pd.read_csv(cfg.analysis_table, usecols=["Entry", "species", "protein_family"])
    assert (len(main), main["species"].nunique(), main["protein_family"].nunique()) == (10148, 495, 281)


def test_supplementary_input_contracts() -> None:
    from decoding_bias.config import Config

    cfg = Config.load()
    sc = pd.read_csv(cfg.finetune_dir / "self_consistency.csv")
    assert len(sc) == 625
    assert sc["uniprot_id"].nunique() == 25
    assert set(sc["model"]) == {
        "ProteinMPNN_v020(base)", "AlkSecMPNN_020", "AcidSecMPNN_020",
        "WT_singleseq(control)",
    }
    pdb = pd.read_csv(cfg.pdb_dir / "cohort_pdb_scored.csv")
    assert (len(pdb), pdb["Entry"].nunique(), pdb["pdb_id"].nunique()) == (876, 876, 876)
    assert pdb["domain"].value_counts().to_dict() == {
        "Eukaryota": 396, "Bacteria": 364, "Archaea": 116,
    }
