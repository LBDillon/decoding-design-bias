"""Reconstruct Figure 3 from a Source Data ZIP or directory, without refitting."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decoding_bias.analysis.figure3 import reconstruct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reconstruct(args.source_data, args.output_dir)
