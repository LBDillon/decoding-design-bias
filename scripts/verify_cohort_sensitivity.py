"""Variance decomposition on the primary, non-ribosomal and domain-balanced cohorts."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoding_bias.analysis import variance  # noqa: E402
from decoding_bias.data import load_analysis_table  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("results/cohort_sensitivity"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

data = load_analysis_table(ROOT / "data/main_analysis.csv", domains_only=True)
data = data.merge(variance.pca_coordinates(data), on="Entry", validate="one_to_one")
scores = [column for column, _ in variance.MODELS.values()]
complete = data.dropna(subset=variance.BIOPHYS + scores)

# Fixed membership: 1,353 proteins per domain, drawn once in R with seed 0.
balanced = pd.read_csv(ROOT / "data/robustness/domain_balanced_membership.csv")
assert balanced.Entry.is_unique and balanced.domain.value_counts().eq(1353).all()

cohorts = {
    "primary": complete,
    "no_ribosomal": complete[complete.broad_function.ne("ribosomal")],
    "domain_balanced": complete[complete.Entry.isin(balanced.Entry)],
}
for name, frame in cohorts.items():
    result = variance.compute(frame)
    result.to_csv(args.output_dir / f"variance_{name}.csv", index=False)
    expected = pd.read_csv(ROOT / f"expected/robustness/variance_{name}.csv").set_index("model").sort_index()
    actual = result.set_index("model").sort_index()
    columns = [column for column in expected.select_dtypes("number") if column in actual]
    assert np.allclose(actual[columns], expected[columns], atol=1e-8, rtol=1e-7), name
    print(f"{name}: {len(frame)} proteins, {len(result)} models; matches expected/robustness")
