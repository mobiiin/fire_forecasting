#!/usr/bin/env bash
#SBATCH --job-name=cawfe_latte_v1_2
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm/cawfe_latte_v1_2_%j.out
#SBATCH --error=artifacts/logs/slurm/cawfe_latte_v1_2_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mhabibp/fire_forecasting}"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/experiments/cawfe_latte_v1_2.yaml}}"
RUN_NAME="${RUN_NAME:-cawfe_latte_v1_2_slurm${SLURM_JOB_ID:-local}}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm artifacts/checkpoints /tmp/mhabibp_mplconfig

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
  echo "Could not find python after activating conda environment ${CONDA_ENV}." >&2
  exit 1
fi

if [[ ":${PYTHONPATH:-}:" != *":${REPO_ROOT}:"* ]]; then
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "========== Slurm =========="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Repo: ${REPO_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Run name: ${RUN_NAME}"
echo "Conda env: ${CONDA_ENV}"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "========== GPU =========="
nvidia-smi

echo "========== CAWFE-Latte v1.2 deep sanity check =========="
srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/sanity_check_project.py \
  --config "${CONFIG_PATH}" \
  --batch_size 2 \
  --num_workers 0 \
  --deep

echo "========== CAWFE-Latte v1.2 one-batch smoke train/val =========="
srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/smoke_test_cawfe_latte_training.py \
  --config "${CONFIG_PATH}" \
  --batch_size 8 \
  --num_workers 0

echo "========== Training =========="
srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
  /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
  scripts/train_forecasting_model.py \
  --config "${CONFIG_PATH}" \
  --run_name "${RUN_NAME}"

BEST_CHECKPOINT="${REPO_ROOT}/artifacts/runs/cawfe_latte_v1_2/${RUN_NAME}/checkpoints/best_model.pt"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-30}"
RUN_QUALITATIVE_EVAL="${RUN_QUALITATIVE_EVAL:-1}"

if [[ "${RUN_QUALITATIVE_EVAL}" == "1" ]]; then
  echo "========== CAWFE-Latte v1.2 qualitative evaluation =========="
  echo "Checkpoint: ${BEST_CHECKPOINT}"
  echo "Split: ${EVAL_SPLIT}"
  echo "Samples: ${EVAL_NUM_SAMPLES}"
  if [[ ! -f "${BEST_CHECKPOINT}" ]]; then
    echo "Best checkpoint not found after training: ${BEST_CHECKPOINT}" >&2
    exit 3
  fi
  srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL \
    /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" \
    scripts/evaluate_trained_models.py \
    --config "${CONFIG_PATH}" \
    --mode qualitative \
    --split "${EVAL_SPLIT}" \
    --model_architecture cawfe_latte_v1_2 \
    --num_samples "${EVAL_NUM_SAMPLES}" \
    --checkpoint "${BEST_CHECKPOINT}" \
    --eval_name "${RUN_NAME}_qualitative_${EVAL_SPLIT}"
else
  echo "Skipping qualitative evaluation because RUN_QUALITATIVE_EVAL=${RUN_QUALITATIVE_EVAL}"
fi
