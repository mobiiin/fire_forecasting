#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: bash $0 <registered-ablation-name> [<registered-ablation-name> ...]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
for ABLATION in "$@"; do
  if ! "${PYTHON_BIN}" scripts/run_cawfe_latte_ablation.py "${ABLATION}" --print-config >/dev/null 2>&1; then
    echo "Unknown ablation: ${ABLATION}" >&2
    exit 2
  fi
  JOB_ID="$(sbatch --parsable scripts/slurm_train_cawfe_latte_ablation_a10080.sh "${ABLATION}")"
  echo "${ABLATION}: submitted job ${JOB_ID}"
done
