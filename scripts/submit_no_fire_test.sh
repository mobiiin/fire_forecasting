#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN=false
MODELS=()
SEEDS=()
EXACT_RUNS=()
RUN_MODELS=()
RUN_SEEDS=()
RUN_KEYS=()

usage() {
  echo "Usage:"
  echo "  bash $0"
  echo "  bash $0 [--model baseline|GA_Q2 ...] [--seed 42|123|2026 ...] [--dry-run]"
  echo "  bash $0 --run MODEL:SEED [--run MODEL:SEED ...] [--dry-run]"
  echo ""
  echo "Default: submit six independent frozen-checkpoint evaluation jobs."
  echo "After they finish: python scripts/evaluate_no_fire_test.py --aggregate-only"
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "$1 requires a value" >&2
    exit 2
  fi
}

add_run() {
  local model="$1"
  local seed="$2"
  local key="${model}:${seed}"
  local existing
  for existing in "${RUN_KEYS[@]}"; do
    if [[ "${existing}" == "${key}" ]]; then
      return
    fi
  done
  RUN_KEYS+=("${key}")
  RUN_MODELS+=("${model}")
  RUN_SEEDS+=("${seed}")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$@"
      MODELS+=("$2")
      shift 2
      ;;
    --seed)
      require_value "$@"
      SEEDS+=("$2")
      shift 2
      ;;
    --run)
      require_value "$@"
      EXACT_RUNS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#EXACT_RUNS[@]} -gt 0 && ( ${#MODELS[@]} -gt 0 || ${#SEEDS[@]} -gt 0 ) ]]; then
  echo "--run cannot be combined with --model or --seed" >&2
  exit 2
fi

mapfile -t REGISTERED_MODELS < <("${PYTHON_BIN}" scripts/evaluate_no_fire_test.py --list-models)
mapfile -t REGISTERED_SEEDS < <("${PYTHON_BIN}" scripts/evaluate_no_fire_test.py --list-seeds)

if [[ ${#EXACT_RUNS[@]} -gt 0 ]]; then
  for spec in "${EXACT_RUNS[@]}"; do
    model="${spec%%:*}"
    seed="${spec#*:}"
    if [[ "${spec}" != *:* || -z "${model}" || -z "${seed}" || "${seed}" == *:* ]]; then
      echo "Invalid --run value ${spec}; expected MODEL:SEED" >&2
      exit 2
    fi
    add_run "${model}" "${seed}"
  done
else
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("${REGISTERED_MODELS[@]}")
  fi
  if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=("${REGISTERED_SEEDS[@]}")
  fi
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      add_run "${model}" "${seed}"
    done
  done
fi

for index in "${!RUN_MODELS[@]}"; do
  model="${RUN_MODELS[$index]}"
  seed="${RUN_SEEDS[$index]}"
  model_ok=false
  seed_ok=false
  for registered in "${REGISTERED_MODELS[@]}"; do
    [[ "${model}" == "${registered}" ]] && model_ok=true
  done
  for registered in "${REGISTERED_SEEDS[@]}"; do
    [[ "${seed}" == "${registered}" ]] && seed_ok=true
  done
  if [[ "${model_ok}" != true ]]; then
    echo "Unknown no-fire model: ${model}" >&2
    exit 2
  fi
  if [[ "${seed_ok}" != true ]]; then
    echo "Unknown no-fire seed: ${seed}" >&2
    exit 2
  fi
done

echo "Selected no-fire test jobs (${#RUN_MODELS[@]}):"
for index in "${!RUN_MODELS[@]}"; do
  echo "  ${RUN_MODELS[$index]} seed=${RUN_SEEDS[$index]}"
done

if [[ "${DRY_RUN}" == true ]]; then
  echo "Dry run: no Slurm jobs were submitted."
  for index in "${!RUN_MODELS[@]}"; do
    echo "  ${SBATCH_BIN} --parsable scripts/slurm_evaluate_no_fire_test_a10080.sh ${RUN_MODELS[$index]} ${RUN_SEEDS[$index]}"
  done
  exit 0
fi

for index in "${!RUN_MODELS[@]}"; do
  model="${RUN_MODELS[$index]}"
  seed="${RUN_SEEDS[$index]}"
  job_id="$("${SBATCH_BIN}" --parsable scripts/slurm_evaluate_no_fire_test_a10080.sh "${model}" "${seed}")"
  echo "${model} seed=${seed} -> job ${job_id}"
done

echo "After all six selected jobs finish, aggregate with:"
echo "  ${PYTHON_BIN} scripts/evaluate_no_fire_test.py --aggregate-only"
