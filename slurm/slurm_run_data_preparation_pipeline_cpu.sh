#!/usr/bin/env bash
#SBATCH --job-name=data_prep_cpu
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --time=72:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm/data_prep_cpu_%j.out
#SBATCH --error=artifacts/logs/slurm/data_prep_cpu_%j.err

set -euo pipefail

# CPU-only full data-preparation job.
# This intentionally rebuilds/overwrites processed outputs in the configured
# scratch dataset root. It does not request a GPU and it does not pass any
# skip-existing options.

REPO_ROOT="${REPO_ROOT:-/home/mhabibp/fire_forecasting}"
CONFIG_PATH="${CONFIG_PATH:-${1:-configs/default.yaml}}"
DERIVED_CONFIG_PATH="${DERIVED_CONFIG_PATH:-}"
PERCENTILE="${PERCENTILE:-5.0}"
PATTERN="${PATTERN:-sparse5_h10}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"
MAKE_QUICKLOOKS="${MAKE_QUICKLOOKS:-0}"
MAX_FRAMES_PER_FIRE="${MAX_FRAMES_PER_FIRE:-}"
# Eight workers gives independent frame compression/feature work enough room to
# overlap with the shared scratch filesystem while avoiding I/O saturation.
ENGINEERED_WORKERS="${ENGINEERED_WORKERS:-8}"
# The normalization stage revisits the same compressed frames for every patch.
# This cache is bounded and uses the job's otherwise idle RAM.
NORMALIZATION_FRAME_CACHE_GB="${NORMALIZATION_FRAME_CACHE_GB:-96}"

cd "${REPO_ROOT}"
mkdir -p artifacts/logs/slurm /tmp/mhabibp_mplconfig configs/derived

export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export FIRE_FORECASTING_PROGRESS_BAR=1
export FIRE_FORECASTING_PROGRESS_PERCENT=5

# Keep threaded numeric libraries from oversubscribing the CPU allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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

PIPELINE_ARGS=("${CONFIG_PATH}" --overwrite --percentile "${PERCENTILE}" --pattern "${PATTERN}" --all-patterns --engineered-workers "${ENGINEERED_WORKERS}" --normalization-frame-cache-gb "${NORMALIZATION_FRAME_CACHE_GB}")
if [[ -n "${DERIVED_CONFIG_PATH}" ]]; then
  PIPELINE_ARGS+=(--derived-config "${DERIVED_CONFIG_PATH}")
fi
if [[ "${MAKE_QUICKLOOKS}" == "1" ]]; then
  PIPELINE_ARGS+=(--make-quicklooks)
fi
if [[ -n "${MAX_FRAMES_PER_FIRE}" ]]; then
  PIPELINE_ARGS+=(--max-frames-per-fire "${MAX_FRAMES_PER_FIRE}")
fi

echo "========== Slurm data preparation =========="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Repo: ${REPO_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Derived config override: ${DERIVED_CONFIG_PATH:-<auto>}"
echo "Percentile: ${PERCENTILE}"
echo "Normalization pattern: ${PATTERN}"
echo "Engineered-frame workers: ${ENGINEERED_WORKERS}"
echo "Normalization frame cache (GiB): ${NORMALIZATION_FRAME_CACHE_GB}"
echo "Conda env: ${CONDA_ENV}"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU request: none"

echo "========== Command =========="
printf 'bash scripts/run_data_preparation_pipeline.sh'
printf ' %q' "${PIPELINE_ARGS[@]}"
echo

echo "========== Running full CPU data preparation =========="
srun --ntasks=1 --chdir="${REPO_ROOT}" --export=ALL   /usr/bin/env PYTHONPATH="${PYTHONPATH}" bash scripts/run_data_preparation_pipeline.sh "${PIPELINE_ARGS[@]}"

echo "========== Done =========="
echo "Data preparation Slurm job completed successfully."
