#!/usr/bin/env bash
#SBATCH --job-name=eval_models
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm_evaluate_trained_models_%j.out
#SBATCH --error=artifacts/logs/slurm_evaluate_trained_models_%j.err

set -euo pipefail

REPO_ROOT="/home/mhabibp/fire_forecasting"
cd "${REPO_ROOT}"

mkdir -p artifacts/logs artifacts/results /tmp/mhabibp_mplconfig

export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export FIRE_FORECASTING_PROGRESS_BAR=0
export FIRE_FORECASTING_PROGRESS_PERCENT=5
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

if [[ ":${PYTHONPATH:-}:" != *":${REPO_ROOT}:"* ]]; then
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${PYTHONPATH:-}"
fi

# Edit these values to evaluate a different architecture, mode, split, or run.
CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
MODE="${MODE:-qualitative}"
SPLIT="${SPLIT:-test}"
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-convlstm_unet}"
RUN_NAME="${RUN_NAME:-REPLACE_WITH_RUN_NAME}"
PAPER_ENERGY_METRIC="${PAPER_ENERGY_METRIC:-log}"
CHECKPOINT="${CHECKPOINT:-artifacts/runs/convlstm_unet/convlstm_unet_14856986/checkpoints/best_model.pt}"

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ "${RUN_NAME}" == "REPLACE_WITH_RUN_NAME" ]]; then
    echo "Set RUN_NAME near the top of this script, or set CHECKPOINT to an explicit checkpoint path." >&2
    exit 2
  fi
  CHECKPOINT="artifacts/runs/${MODEL_ARCHITECTURE}/${RUN_NAME}/checkpoints/best_model.pt"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

echo "Python executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "Working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

echo "========== Slurm =========="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory per node MB: ${SLURM_MEM_PER_NODE:-unknown}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "========== GPU =========="
nvidia-smi

echo "========== Evaluation =========="
echo "Config: ${CONFIG_PATH}"
echo "Mode: ${MODE}"
echo "Split: ${SPLIT}"
echo "Model architecture: ${MODEL_ARCHITECTURE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Paper energy metric: ${PAPER_ENERGY_METRIC}"

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" scripts/evaluate_trained_models.py \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --split "${SPLIT}" \
  --model_architecture "${MODEL_ARCHITECTURE}" \
  --checkpoint "${CHECKPOINT}" \
  --paper_energy_metric "${PAPER_ENERGY_METRIC}"
