"""The quick reproduction, including the generated importance views."""
import shutil

import pandas as pd

from decoding_bias.config import Config
from decoding_bias.reproduce import run, _compare_importance_views


def test_quick_reproduction(tmp_path_factory, tmp_path):
    cfg = Config.load(output_dir=tmp_path_factory.mktemp("reproduction"))
    ok, report = run(cfg, quick=True)
    assert ok
    assert not (report["status"] == "FAIL").any()
    assert (cfg.output_dir / "reproduction/reproduction_report.md").exists()
    # A relabelled model column in a generated view must be detected.
    reference = cfg.expected_dir / "importance/property_importance_expanded.csv"
    for name in ["physchem_relweight_wide_expanded.csv", "physchem_relweight_wide_finetune020.csv"]:
        shutil.copy2(cfg.output_dir / "reproduction/importance" / name, tmp_path / name)
    assert _compare_importance_views(reference, tmp_path)[0]
    path = tmp_path / "physchem_relweight_wide_expanded.csv"
    view = pd.read_csv(path)
    view.rename(columns={view.columns[1]: "wrong-model"}).to_csv(path, index=False)
    assert not _compare_importance_views(reference, tmp_path)[0]
