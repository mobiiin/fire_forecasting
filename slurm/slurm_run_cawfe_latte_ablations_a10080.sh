#!/usr/bin/env bash
#SBATCH --job-name=cawfe_latte_ablations
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
#SBATCH --output=artifacts/logs/slurm/cawfe_latte_ablations_%j.out
#SBATCH --error=artifacts/logs/slurm/cawfe_latte_ablations_%j.err
set -euo pipefail
ABLATIONS="${1:-baseline A_resblocks_only B1_softplus_only B2_support_gate_mask_target C_temporal_attention_only}"
MODE="${2:-fast}"
REPO_ROOT="${REPO_ROOT:-/home/mhabibp/fire_forecasting}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fire_forecasting}"
cd "$REPO_ROOT"; mkdir -p artifacts/logs/slurm /tmp/mhabibp_mplconfig
export MPLCONFIGDIR=/tmp/mhabibp_mplconfig PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/run_cawfe_latte_ablations.py --ablations $ABLATIONS --mode "$MODE"
