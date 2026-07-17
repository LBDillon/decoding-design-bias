"""Command-line interface for the compact reviewer reproduction."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from .config import Config


def _config(args) -> Config:
    return Config.load(data=getattr(args, "data", None), output_dir=getattr(args, "output_dir", None))


def cmd_reproduce(args) -> int:
    from .reproduce import run
    ok, _ = run(_config(args), quick=args.quick)
    return 0 if ok else 1


def cmd_dataset(args) -> int:
    from .data import dataset_composition, load_analysis_table
    result = dataset_composition(load_analysis_table(_config(args).analysis_table))
    print(f"{result['n_proteins']} proteins / {result['n_species']} species / "
          f"{result['n_families']} families")
    return 0


def cmd_variance(args) -> int:
    from .analysis import variance
    cfg = _config(args)
    variance.run(cfg, make_figures=not args.no_figures)
    variance.run_plddt(cfg)
    return 0


def cmd_taxonomy(args) -> int:
    from .analysis import elo
    elo.run(_config(args), n_permutations=args.permutations)
    return 0


def cmd_pca(args) -> int:
    from .analysis import pca
    pca.run(_config(args))
    return 0


def cmd_gam(args) -> int:
    cfg = _config(args)
    if not shutil.which("Rscript"):
        print("Rscript was not found; see environment-R.md", file=sys.stderr)
        return 2
    script = cfg.root / "src/decoding_bias/analysis/gam_landscapes.R"
    completed = subprocess.run(["Rscript", str(script), str(cfg.analysis_table), str(cfg.stage_output("gam"))])
    return completed.returncode


def cmd_importance(args) -> int:
    from .analysis import importance
    importance.run(_config(args))
    return 0


def cmd_design(args) -> int:
    from .analysis import design
    design.run(_config(args))
    return 0


def cmd_finetune(args) -> int:
    from .analysis import finetune
    finetune.run(_config(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decoding-bias",
        description="Reproduce the quantitative results in Decoding design bias.",
    )
    parser.add_argument("--data", help="override data/main_analysis.csv")
    parser.add_argument("--output-dir", help="override the results directory")
    commands = parser.add_subparsers(dest="command", required=True)

    reproduce = commands.add_parser("reproduce", help="run and verify the reviewer reproduction")
    reproduce.add_argument("--quick", action="store_true", help="skip Elo and R/GAM")
    reproduce.set_defaults(function=cmd_reproduce)

    dataset = commands.add_parser("dataset", help="print the deposited cohort composition")
    dataset.set_defaults(function=cmd_dataset)

    variance = commands.add_parser("variance", help="Tables 1-2 and pLDDT control")
    variance.add_argument("--no-figures", action="store_true")
    variance.set_defaults(function=cmd_variance)

    taxonomy = commands.add_parser("taxonomy", help="Figure 2 and Table 3 species Elo")
    taxonomy.add_argument("--permutations", type=int, default=50)
    taxonomy.set_defaults(function=cmd_taxonomy)

    for name, help_text, function in [
        ("pca", "Figure 3A-B and Table S14", cmd_pca),
        ("gam", "Figure 3C and Table S15 (R/mgcv)", cmd_gam),
        ("importance", "Tables S16-S18", cmd_importance),
        ("design", "Figure 4 and Table 4", cmd_design),
        ("finetune", "Figure 5 and Tables 5-6", cmd_finetune),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(function=function)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
