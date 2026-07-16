"""decoding-bias command-line interface.

  dataset      validate the analysis table + reproduce the composition (Table 7)
  score        run/import model likelihood scores            [needs weights]
  variance     score-variance decomposition (Table 1/2, SI Fig S2, Tables S9/S10)
  taxonomy     species Elo taxonomy preference (Fig 2, Table 3, SI S5-S8)
  pca          biophysical PCA tables (Fig 3A/B, Table S14, compactness)  [+R for GAM]
  importance   property-to-score importance (SI Fig S6, Tables S16-S18)
  design       designed-sequence analysis (Fig 4, Tables 4-6)  [needs design_dir]
  finetune     fine-tuning arms (Fig 5, SI Figs S8-S11)         [needs weights]
  pdb-cohort   experimental-PDB cohort (SI Figs S3-S4)          [needs structures]
  figures      status of every paper/SI output + run the reproducible ones
  verify       re-run reproducible stages and diff vs tests/reference/

Paths and parameters resolve through decoding_bias.config; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import sys

from .config import Config


def _cfg(args) -> Config:
    return Config.load(config_path=getattr(args, "config", None),
                       data=getattr(args, "data", None),
                       output_dir=getattr(args, "output_dir", None))


def _blocked(stage: str, needs: str, config_key: str, real_commands: list[str]) -> int:
    print(f"[{stage}] Unavailable: this stage requires {needs}, which is not part of the deposit.")
    print(f"          Set paths.{config_key} in a config YAML (or DECODING_BIAS_PATHS_"
          f"{config_key.upper()}) and rerun.")
    print("          Canonical commands this stage runs once inputs are available:")
    for c in real_commands:
        print(f"            $ {c}")
    return 0


# ----------------------------- stage handlers ----------------------------- #
def cmd_dataset(args):
    cfg = _cfg(args)
    from .data import load_analysis_table, dataset_composition
    import pandas as pd
    df = load_analysis_table(cfg.analysis_table)
    comp = dataset_composition(df)
    out = cfg.stage_output("dataset")
    rows = [{"metric": "n_proteins", "value": comp["n_proteins"]},
            {"metric": "n_species", "value": comp["n_species"]},
            {"metric": "n_families", "value": comp["n_families"]},
            {"metric": "families_three_domain", "value": comp["families_three_domain"]},
            {"metric": "families_two_domain", "value": comp["families_two_domain"]},
            {"metric": "families_single_domain", "value": comp["families_single_domain"]}]
    for dom, d in comp["by_domain"].items():
        rows += [{"metric": f"{dom}_proteins", "value": d["proteins"]},
                 {"metric": f"{dom}_proteins_pct", "value": d["proteins_pct"]},
                 {"metric": f"{dom}_species", "value": d["species"]}]
    pd.DataFrame(rows).to_csv(out / "composition.csv", index=False)
    print(f"[dataset] {comp['n_proteins']} proteins / {comp['n_species']} species / "
          f"{comp['n_families']} families  ->  {out/'composition.csv'}")
    return 0


def cmd_variance(args):
    cfg = _cfg(args)
    from .analysis import variance
    if getattr(args, "plddt", False):
        variance.run_plddt(cfg)          # Table S13: decompose avg_plddt (shipped column)
        return 0
    variance.run(cfg, make_figures=not args.no_figures)
    return 0


def cmd_taxonomy(args):
    cfg = _cfg(args)
    from .analysis import elo
    weightings = ("unweighted", "plddt_weighted", "plddt_residual") if args.all_variants else ("unweighted",)
    arms = ("full", "ft020") if args.all_variants else ("full",)
    elo.run(cfg, arms=arms, weightings=weightings, make_paper_figures=not args.no_figures)
    return 0


def cmd_pca(args):
    cfg = _cfg(args)
    from .analysis import pca
    pca.run(cfg)
    if getattr(args, "gam_deviance", False) or getattr(args, "gam_landscapes", False):
        print("[pca] GAM deviance/landscapes require R + mgcv via the notebook "
              "04_pca_gam/PCA_paper_figures.ipynb (see external-services.md).")
    return 0


def cmd_importance(args):
    cfg = _cfg(args)
    from .analysis import importance
    importance.run(cfg)
    return 0


def cmd_score(args):
    return _blocked("score", "model weights (per-model likelihood scoring)", "weights_dir",
                    ["# scores already ship in the analysis table; to recompute:",
                     "# Colab per-model notebooks in notebooks/01_scoring/ + decoding_bias/scoring/",
                     "#   (model_score_registry.yaml, score_esm2_arc.py)"])


def cmd_design(args):
    cfg = _cfg(args)
    if cfg.external("design_dir") is not None:
        from .analysis import design
        res = design.run(cfg)     # shift (Fig 4/S21) + functional residue (Table 4/S20)
        if res["functional"] is None:
            print("[design] (functional-residue recovery skipped: no all_designs_and_wt.csv in design_dir)")
        print("[design] design ΔTm still needs a DeepStabP predicted-Tm input.")
        return 0
    return _blocked("design", "the design tables (designs_features.csv + wt_features.csv for the "
                    "shifts; all_designs_and_wt.csv + designs_ph_features.csv + optional "
                    "_uniprot_features_cache.json for functional-residue recovery)", "design_dir",
                    ["decoding-bias design    # runs Fig 4/S21 + Table 4/S20 once paths.design_dir is set"])


def cmd_finetune(args):
    cfg = _cfg(args)
    from .analysis import finetune
    try:
        finetune.table_s22(cfg)     # Table S22 from the deposited matched-surface steers
    except FileNotFoundError as e:
        print(f"[finetune] Table S22 unavailable: {e}")
    print("[finetune] The full fine-tuning + generation arms (Fig 5, SI Figs S8-S11) need the "
          "fine-tuned weights + GPU: see decoding_bias/finetune/ (ProteinMPNN arm) and "
          "train_esm2_mlm.py / notebooks/07_finetuning/ (ESM2-35M arm).")
    return 0


def cmd_pdb_cohort(args):
    return _blocked("pdb-cohort", "experimental PDB structures from RCSB", "structures_dir",
                    ["# build cohort + matched control: decoding_bias/pdb_cohort/",
                     "#   build_independent_pdb_cohort.py -> run_matched_af2_control.py -> compare_cohort_to_main.py",
                     "#   (needs the RCSB Search API + downloaded structures)"])


def cmd_figures(args):
    from . import manifest
    from .config import repo_root
    root = repo_root()
    print(f"{'id':12} {'label':16} {'status':8} {'out':5} stage / note")
    print("-" * 92)
    for o in manifest.OUTPUTS:
        exists = any((root / p).exists() for p in o["outputs"]) if o["outputs"] else False
        print(f"{o['id']:12} {o['label']:16} {o['status']:8} {'yes' if exists else '-':5} "
              f"{o['stage']}  {o.get('note','')[:48]}")
    counts = {}
    for o in manifest.OUTPUTS:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    print("\nstatus summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.run:
        cfg = _cfg(args)
        from .analysis import variance, importance, pca
        print("\nrunning reproducible stages -> results/ ...")
        cmd_dataset(args)
        variance.run(cfg)
        importance.run(cfg)
        pca.run(cfg)
        print("Elo (Fig 2 / Table 3) is slow (~7 min); run `decoding-bias taxonomy` separately.")
    return 0


def cmd_verify(args):
    cfg = _cfg(args)
    from . import verify
    ok = verify.run(cfg, fast=not args.full)
    return 0 if ok else 1


# ------------------------------- parser ----------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decoding-bias", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="user config YAML (overrides defaults)")
    p.add_argument("--data", help="analysis table path (overrides config)")
    p.add_argument("--output-dir", dest="output_dir", help="output root (overrides config)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("dataset", help="validate table + composition (Table 7)")
    sp.set_defaults(func=cmd_dataset)

    sp = sub.add_parser("variance", help="variance decomposition (Table 1/2)")
    sp.add_argument("--no-figures", action="store_true")
    sp.add_argument("--plddt", action="store_true", help="pLDDT-as-score decomposition (needs metadata)")
    sp.set_defaults(func=cmd_variance)

    sp = sub.add_parser("taxonomy", help="species Elo (Fig 2, Table 3)")
    sp.add_argument("--all-variants", action="store_true",
                    help="all weighting schemes + FT arm (Tables S5-S8, post-FT Elo)")
    sp.add_argument("--no-figures", action="store_true")
    sp.set_defaults(func=cmd_taxonomy)

    sp = sub.add_parser("pca", help="biophysical PCA tables (Fig 3A/B, Table S14)")
    sp.add_argument("--gam-deviance", action="store_true")
    sp.add_argument("--gam-landscapes", action="store_true")
    sp.set_defaults(func=cmd_pca)

    sp = sub.add_parser("importance", help="property importance (SI S6, S16-S18)")
    sp.set_defaults(func=cmd_importance)

    sp = sub.add_parser("score", help="model likelihood scoring [needs weights]")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("design", help="designed-sequence analysis (Fig 4) [needs design_dir]")
    sp.set_defaults(func=cmd_design)

    sp = sub.add_parser("finetune", help="fine-tuning arms (Fig 5) [needs weights]")
    sp.set_defaults(func=cmd_finetune)

    sp = sub.add_parser("pdb-cohort", help="experimental-PDB cohort (SI S5) [needs structures]")
    sp.set_defaults(func=cmd_pdb_cohort)

    sp = sub.add_parser("figures", help="status of all paper outputs; --run the reproducible ones")
    sp.add_argument("--run", action="store_true")
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("verify", help="re-run reproducible stages, diff vs tests/reference/")
    sp.add_argument("--full", action="store_true", help="include Elo (slow)")
    sp.set_defaults(func=cmd_verify)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
