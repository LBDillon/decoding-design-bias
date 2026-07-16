"""Fine-tuning surface-steer analysis (paper Table S22 / Fig S10).

The full fine-tuning + generation pipeline (`decoding_bias/finetune/`) needs GPU +
model weights. But the matched surface-only steer per target:
(`00_data/finetune/surface_shift_matched.csv`, from the local ESM2-35M generation
run). Table S22 is a simple per-model aggregation of its `case_minus_control` column
(AlkSec − neutralophile control)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def table_s22(cfg, out_dir: Path | None = None) -> pd.DataFrame:
    src = cfg.root / "00_data" / "finetune" / "surface_shift_matched.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found (deposited matched-surface steers).")
    out_dir = Path(out_dir) if out_dir else cfg.stage_output("finetune")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    rows = []
    for fam, g in df.groupby("family"):
        x = g["case_minus_control"].values
        rows.append(dict(model=fam, mean_steer=x.mean(),
                         sem=x.std(ddof=1) / np.sqrt(len(x)),
                         n_targets=len(x), n_acidic=int((x < 0).sum())))
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "table_s22_surface_steer.csv", index=False)
    print(f"[finetune] Table S22 (matched surface steer, AlkSec−control) -> "
          f"{out_dir/'table_s22_surface_steer.csv'}")
    for r in res.itertuples(index=False):
        print(f"    {r.model:11} {r.mean_steer:+.3f} ± {r.sem:.3f}  ({r.n_acidic}/{r.n_targets} acidic)")
    return res
