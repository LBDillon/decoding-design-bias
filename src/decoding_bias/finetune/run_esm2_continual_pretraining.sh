#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/esm35m_continual_pretraining}"
STAGE="${1:-help}"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"/{data,runs,scores,tables,figures,logs}

case "${STAGE}" in
  prep)
    python paper_code/10_esm_continual_pretraining/prepare_esm_secretome_data.py \
      --input_dir finetune/data \
      --out_dir "${OUT_DIR}/data"
    ;;
  env-check)
    sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_env_check.slurm
    ;;
  dry-run)
    sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_dry_run.slurm
    ;;
  train)
    echo "Submit this only after the ARC dry-run job succeeds."
    sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_train_array.slurm
    ;;
  score)
    sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_score_array.slurm
    ;;
  order)
    cat <<'EOF'
Recommended ARC run order:
  sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_env_check.slurm
  python paper_code/10_esm_continual_pretraining/prepare_esm_secretome_data.py \
    --input_dir finetune/data \
    --out_dir outputs/esm35m_continual_pretraining/data
  sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_dry_run.slurm

Only after dry-run success:
  sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_train_array.slurm

After training completes:
  sbatch paper_code/10_esm_continual_pretraining/slurm/esm35m_score_array.slurm
EOF
    ;;
  *)
    cat <<EOF
Usage: $0 {prep|env-check|dry-run|train|score|order}

Environment overrides:
  PROJECT_DIR=${PROJECT_DIR}
  OUT_DIR=${OUT_DIR}
EOF
    ;;
esac
