#!/usr/bin/env bash
#SBATCH --job-name=convlstm_ablation
#SBATCH --array=0-2
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=72:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm_convlstm_ablation_%A_%a.out
#SBATCH --error=artifacts/logs/slurm_convlstm_ablation_%A_%a.err

set -euo pipefail

REPO_ROOT="/home/mhabibp/fire_forecasting"
cd "${REPO_ROOT}"

mkdir -p artifacts/logs artifacts/checkpoints /tmp/mhabibp_mplconfig

export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fire_forecasting
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python after activating conda environment fire_forecasting." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIGS=(
  "configs/experiments/convlstm_consecutive5_h10.yaml"
  "configs/experiments/convlstm_single1_h10.yaml"
  "configs/experiments/convlstm_sparse5_h10.yaml"
)
PATTERNS=(consecutive5_h10 single1_h10 sparse5_h10)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-$(( ${#CONFIGS[@]} - 1 ))" >&2
  exit 2
fi

CONFIG_PATH="${CONFIGS[$TASK_ID]}"
PATTERN="${PATTERNS[$TASK_ID]}"
RUN_NAME="convlstm_${PATTERN}_slurm${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_task${TASK_ID}"

printf '========== Slurm ==========%s\n' ""
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-unknown}/${TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Pattern: ${PATTERN}"
echo "Config: ${CONFIG_PATH}"
echo "Run name: ${RUN_NAME}"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "========== GPU =========="
nvidia-smi

echo "========== Training =========="

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/train_forecasting_model.py \
  --config "${CONFIG_PATH}" \
  --run_name "${RUN_NAME}"
