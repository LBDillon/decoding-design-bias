"""Model identities, accession exclusions and supplementary input tables."""
import tempfile
from pathlib import Path

import pandas as pd

from decoding_bias import catalog
from decoding_bias.config import Config
from decoding_bias.data import load_analysis_table


def test_supplementary_input_contracts() -> None:
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


def test_xl_is_primary_and_base_remains_a_distinct_sensitivity() -> None:
    assert catalog.FULL_COHORT["ProGen2-XL"] == ("progen2_XL_score", "sequence")
    assert len(catalog.FULL_COHORT) == 14
    assert "progen2_score" not in catalog.ELO_FULL_COLUMNS
    assert catalog.IMPORTANCE_PANEL["ProGen2-base"] == "progen2_score"
    assert catalog.IMPORTANCE_PANEL["ProGen2-XL"] == "progen2_XL_score"


def test_primary_loader_excludes_only_the_two_accessions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "input.csv"
        pd.DataFrame({"Entry": ["P19949", "Q8TFG7", "retain"], "domain": ["Archaea"] * 3,
                      "progen2_score": [1., 2., 3.], "progen2_XL_score": [4., 5., 6.]}).to_csv(p, index=False)
        got = load_analysis_table(p)
    assert got.Entry.to_list() == ["retain"]
    assert got.progen2_score.to_list() == [3.]
    assert got.progen2_XL_score.to_list() == [6.]
