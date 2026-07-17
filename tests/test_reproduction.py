"""End-to-end regression test for the compact Python reviewer path."""
from __future__ import annotations

from decoding_bias.config import Config
from decoding_bias.reproduce import run


def test_quick_reproduction(tmp_path) -> None:
    ok, report = run(Config.load(output_dir=tmp_path), quick=True)
    assert ok
    assert not (report["status"] == "FAIL").any()
    assert (tmp_path / "reviewer/reproduction_report.md").exists()
