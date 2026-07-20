#!/usr/bin/env bash
#SBATCH --job-name=ff_diag_convlstm
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm_diagnose_convlstm_%j.out
#SBATCH --error=artifacts/logs/slurm_diagnose_convlstm_%j.err

set -euo pipefail

REPO_ROOT="/home/mhabibp/fire_forecasting"
cd "${REPO_ROOT}"

mkdir -p artifacts/logs /tmp/mhabibp_mplconfig
export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export FIRE_FORECASTING_PROGRESS_BAR=0
export FIRE_FORECASTING_PROGRESS_PERCENT=5
export FIRE_FORECASTING_TIMING_LOG_EVERY_N_BATCHES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fire_forecasting
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python after activating conda environment fire_forecasting." >&2
  exit 1
fi
echo "Python executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
if [[ ":${PYTHONPATH:-}:" != *":${REPO_ROOT}:"* ]]; then
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${PYTHONPATH:-}"
fi
echo "Working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory per node MB: ${SLURM_MEM_PER_NODE:-unknown}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL /usr/bin/env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" -m scripts.diagnose_training_pipeline \
  --config configs/default.yaml \
  --model_architecture convlstm_unet \
  --num_batches 50
