"""Designed-sequence analysis (paper Figure 4, Table 4, Tables S20/S21).

Two independent analyses, each run when its inputs are present in `paths.design_dir`
(the design tables are not part of the deposit; provide them via data-availability):

  shift        physchem_shift_analysis: per-(model, property) WT->design paired
               Cohen's dz. Reproduces Table S21 exactly (verified 1e-16).
               Needs: designs_features.csv, wt_features.csv.
  functional   functional_residue_conservation: WT-residue recovery at annotated
               UniProt functional sites vs background, per model. Reproduces
               Table 4 / S20 exactly (MIF-ST Δ=-0.158, p=1.8e-4).
               Needs: all_designs_and_wt.csv (design + WT sequences),
               designs_ph_features.csv (fine-tuned arms), and either a committed
               _uniprot_features_cache.json or internet (fetches from rest.uniprot.org).

Design ΔTm (DeepStabP) is separate and still needs its own predicted-Tm input.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def _shift(design_dir: Path, out_dir: Path):
    from ..design import physchem_shift_analysis as ps
    eff, _, _ = ps.run(designs_csv=design_dir / "designs_features.csv",
                       wt_csv=design_dir / "wt_features.csv", out_dir=out_dir)
    print(f"[design] shift (Fig 4 / Table S21): {eff['model'].nunique()} models x "
          f"{eff['property'].nunique()} properties")
    return eff


def _functional(design_dir: Path, out_dir: Path):
    from ..design import functional_residue_conservation as fr
    fr.OUT = out_dir
    fr.ALL = design_dir / "all_designs_and_wt.csv"
    fr.FT_DESIGNS = design_dir / "designs_ph_features.csv"
    # copy any committed UniProt cache into out_dir so the source file isn't rewritten
    src_cache = design_dir / "_uniprot_features_cache.json"
    fr.CACHE = out_dir / "_uniprot_features_cache.json"
    if src_cache.exists():
        shutil.copy2(src_cache, fr.CACHE)
    fr.main()
    summary = pd.read_csv(out_dir / "functional_residue_conservation_by_model.csv")
    print(f"[design] functional-residue (Table 4 / S20): {len(summary)} models")
    return summary


def run(cfg, out_dir: Path | None = None) -> dict:
    design_dir = cfg.require("design_dir", "the designed-sequence analysis (Fig 4, Table 4, S20/S21)")
    out_dir = Path(out_dir) if out_dir else cfg.stage_output("design")
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"shift": None, "functional": None}
    if (design_dir / "designs_features.csv").exists() and (design_dir / "wt_features.csv").exists():
        res["shift"] = _shift(design_dir, out_dir)
    if (design_dir / "all_designs_and_wt.csv").exists():
        res["functional"] = _functional(design_dir, out_dir)
    if res["shift"] is None and res["functional"] is None:
        raise FileNotFoundError(
            f"design_dir={design_dir} has neither the shift inputs (designs_features.csv + "
            f"wt_features.csv) nor the functional-residue input (all_designs_and_wt.csv).")
    return res
