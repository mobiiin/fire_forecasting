#!/usr/bin/env bash
#SBATCH --job-name=temp_sparse5_42
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
#SBATCH --output=artifacts/logs/slurm/temp_sparse5_42_%j.out
#SBATCH --error=artifacts/logs/slurm/temp_sparse5_42_%j.err

set -euo pipefail

MODE="sparse5"
SEED="42"
REPO_ROOT="${REPO_ROOT:-/home/mhabibp/fire_forecasting}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_mplconfig
export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export FIRE_FORECASTING_PROGRESS_BAR=1
export FIRE_FORECASTING_PROGRESS_PERCENT=0
export FIRE_FORECASTING_TIMING_LOG_EVERY_N_BATCHES=50
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python after activating ${CONDA_ENV}." >&2
  exit 1
fi
if [[ ":${PYTHONPATH:-}:" != *":${REPO_ROOT}:"* ]]; then
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

CONFIG_PATH="$("${PYTHON_BIN}" scripts/run_temporal_context_ablation.py "${MODE}" "${SEED}" --print-config)"
OUTPUT_PATH="$("${PYTHON_BIN}" scripts/run_temporal_context_ablation.py "${MODE}" "${SEED}" --print-output)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Mode: ${MODE}"
echo "Seed: ${SEED}"
echo "Prepared config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Budget: exactly 10 epochs"
"${PYTHON_BIN}" --version
nvidia-smi

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/run_temporal_context_ablation.py "${MODE}" "${SEED}"
