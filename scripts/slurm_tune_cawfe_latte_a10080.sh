#!/usr/bin/env bash
#SBATCH --job-name=ff_latte_tune
#SBATCH --account=cuuser_fafghah_trajectory_planning_in_unmanned_aerial_veh
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160gb
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/mhabibp/fire_forecasting
#SBATCH --output=artifacts/logs/slurm_tune_cawfe_latte_%j.out
#SBATCH --error=artifacts/logs/slurm_tune_cawfe_latte_%j.err

set -euo pipefail

mkdir -p artifacts/logs artifacts/hparam/cawfe_latte /tmp/mhabibp_mplconfig

export MPLCONFIGDIR=/tmp/mhabibp_mplconfig
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fire_forecasting

echo "========== Slurm =========="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory per node MB: ${SLURM_MEM_PER_NODE:-unknown}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "========== GPU =========="
nvidia-smi

echo "========== Tuning CAWFE-Latte =========="
srun --ntasks=1 python scripts/tune_cawfe_latte.py \
  --config configs/default.yaml \
  --num_trials 12 \
  --output_dir artifacts/hparam/cawfe_latte \
  --method random \
  --seed 42 \
  --resume

echo "========== Applying best tuned params =========="
srun --ntasks=1 python scripts/apply_cawfe_latte_tuned_params.py \
  --base_config configs/default.yaml \
  --best_params artifacts/hparam/cawfe_latte/best_params.json \
  --output_config artifacts/hparam/cawfe_latte/best_config.yaml

echo "Best tuned params:"
cat artifacts/hparam/cawfe_latte/best_params.json
