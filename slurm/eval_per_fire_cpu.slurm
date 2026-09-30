#!/usr/bin/env bash
#SBATCH --job-name=flare_per_fire
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm/flare_per_fire_%j.out
#SBATCH --error=artifacts/logs/slurm/flare_per_fire_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/mhabibp/fire_forecasting}"
CONDA_ENV="${CONDA_ENV:-fire_forecasting}"
MODELS="${MODELS:-persistence,cawfe_st_mamba,cawfe_latte_baseline,cawfe_latte_final}"
SEEDS="${SEEDS:-42,123,2026}"
OUTPUT_DIR="${OUTPUT_DIR:-results/per_fire_generalization}"
cd "${REPO_ROOT}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p artifacts/logs/slurm
args=(--models "${MODELS}" --seeds "${SEEDS}" --output-dir "${OUTPUT_DIR}")
if [[ "${RECOMPUTE:-0}" == "1" ]]; then args+=(--recompute); fi
python scripts/evaluate_per_fire.py "${args[@]}" "$@"
