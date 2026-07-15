#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/esm35m_continual_pretraining}"
ENV_ACTIVATE="${ESM35M_ENV_ACTIVATE:-${HOME}/venvs/esm35m_ft/bin/activate}"

if [[ -f "${ENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"/{data,runs,scores,tables,figures,logs}

python paper_code/10_esm_continual_pretraining/prepare_esm_secretome_data.py \
  --input_dir finetune/data \
  --out_dir "${OUT_DIR}/data"

python paper_code/10_esm_continual_pretraining/train_esm2_mlm.py \
  --train_csv "${OUT_DIR}/data/alkaline_case_train.csv" \
  --val_csv "${OUT_DIR}/data/alkaline_case_val.csv" \
  --out_dir "${OUT_DIR}/runs/dry_run/AlkSecESM35M" \
  --dry_run \
  --overwrite_output_dir

python paper_code/10_esm_continual_pretraining/score_esm2_masked_marginals.py \
  --model_dir facebook/esm2_t12_35M_UR50D \
  --model_name BaseESM35M \
  --self_test \
  --out_csv "${OUT_DIR}/scores/dry_run_BaseESM35M_selftest.csv" \
  --batch_masked_positions 16

python paper_code/10_esm_continual_pretraining/score_esm2_masked_marginals.py \
  --model_dir "${OUT_DIR}/runs/dry_run/AlkSecESM35M" \
  --model_name AlkSecESM35M_dry_run \
  --self_test \
  --out_csv "${OUT_DIR}/scores/dry_run_AlkSecESM35M_selftest.csv" \
  --batch_masked_positions 16

echo "Dry-run training and self-test scoring completed under ${OUT_DIR}"
