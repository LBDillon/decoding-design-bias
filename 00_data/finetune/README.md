# Deposited fine-tuning surface-steer data (Table S22)

`surface_shift_matched.csv` - the per-target matched surface-only redesign steers for
the two models compared in Table S22 (ProteinMPNN vs ESM2-35M), one row per
(family, target). Columns include `case_minus_control` (AlkSec − neutralophile
control; more negative = more acidic surface), which Table S22 aggregates.

Source: the local ESM2-35M run,
`decoding-design-bias/outputs/esm35m_continual_pretraining/generation/surface_shift_matched.csv`
(written by `compare_surface_designs.py`). The full fine-tuning + generation pipeline
that produces it (`decoding_bias/finetune/`) needs the fine-tuned weights
(`runs_local/AlkSecESM35M_e30/`, `NeuSecESM35M_e30/`) + GPU and is blocked in this repo,
so the small per-target result is deposited instead - `decoding-bias finetune` aggregates
it into Table S22.

Reproduces the paper exactly: ProteinMPNN −0.232 ± 0.037 (10/10 targets steered acidic);
ESM2-35M −0.075 ± 0.021 (9/10). The structure model steers ≈3× more on the identical task.
