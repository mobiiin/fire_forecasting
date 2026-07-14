#!/usr/bin/env bash
set -euo pipefail

jid_tune=$(sbatch --parsable slurm/slurm_tune_cawfe_latte_a10080.sh)
echo "Submitted tuning job: ${jid_tune}"

jid_train=$(sbatch --parsable --dependency=afterok:${jid_tune} slurm/slurm_train_cawfe_latte_tuned_a10080.sh)
echo "Submitted training job: ${jid_train}"

jid_ablate=$(sbatch --parsable --dependency=afterok:${jid_train} slurm/slurm_ablate_cawfe_latte_a10080.sh)
echo "Submitted ablation job: ${jid_ablate}"

echo "Pipeline submitted:"
echo "  tune:  ${jid_tune}"
echo "  train: ${jid_train}"
echo "  ablate:${jid_ablate}"
