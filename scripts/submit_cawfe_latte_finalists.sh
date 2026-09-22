#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN=false
FINALISTS=()
SEEDS=()
EXACT_RUNS=()
RUN_FINALISTS=()
RUN_SEEDS=()
RUN_KEYS=()

usage() {
  echo "Usage:"
  echo "  bash $0"
  echo "  bash $0 [--finalist NAME ...] [--seed SEED ...] [--dry-run]"
  echo "  bash $0 --run NAME:SEED [--run NAME:SEED ...] [--dry-run]"
  echo ""
  echo "Selection rules:"
  echo "  No filters                 submit every registered finalist and seed"
  echo "  --finalist/--seed          submit their Cartesian product; both repeat"
  echo "  --run NAME:SEED            submit only exact pairs; repeat as needed"
  echo "  positional finalist names remain supported for compatibility"
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "$1 requires a value" >&2
    exit 2
  fi
}

add_run() {
  local finalist="$1"
  local seed="$2"
  local key="${finalist}:${seed}"
  local existing
  for existing in "${RUN_KEYS[@]}"; do
    if [[ "${existing}" == "${key}" ]]; then
      return
    fi
  done
  RUN_KEYS+=("${key}")
  RUN_FINALISTS+=("${finalist}")
  RUN_SEEDS+=("${seed}")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --finalist)
      require_value "$@"
      FINALISTS+=("$2")
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
    --)
      shift
      while [[ $# -gt 0 ]]; do
        FINALISTS+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      FINALISTS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#EXACT_RUNS[@]} -gt 0 && ( ${#FINALISTS[@]} -gt 0 || ${#SEEDS[@]} -gt 0 ) ]]; then
  echo "--run cannot be combined with --finalist, --seed, or positional finalist names" >&2
  exit 2
fi

if [[ ${#EXACT_RUNS[@]} -gt 0 ]]; then
  for run_spec in "${EXACT_RUNS[@]}"; do
    run_finalist="${run_spec%%:*}"
    run_seed="${run_spec#*:}"
    if [[ "${run_spec}" != *:* || -z "${run_finalist}" || -z "${run_seed}" || "${run_seed}" == *:* ]]; then
      echo "Invalid --run value ${run_spec}; expected NAME:SEED" >&2
      exit 2
    fi
    add_run "${run_finalist}" "${run_seed}"
  done
else
  if [[ ${#FINALISTS[@]} -eq 0 ]]; then
    mapfile -t FINALISTS < <("${PYTHON_BIN}" scripts/run_cawfe_latte_finalist.py --list-finalists)
  fi
  if [[ ${#SEEDS[@]} -eq 0 ]]; then
    mapfile -t SEEDS < <("${PYTHON_BIN}" scripts/run_cawfe_latte_finalist.py --list-seeds)
  fi
  for finalist in "${FINALISTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      add_run "${finalist}" "${seed}"
    done
  done
fi

for index in "${!RUN_FINALISTS[@]}"; do
  "${PYTHON_BIN}" scripts/run_cawfe_latte_finalist.py     "${RUN_FINALISTS[$index]}" "${RUN_SEEDS[$index]}" --print-output-parent >/dev/null
done

"${PYTHON_BIN}" scripts/check_cawfe_latte_finalist_configs.py

echo "Selected finalist jobs (${#RUN_FINALISTS[@]}):"
for index in "${!RUN_FINALISTS[@]}"; do
  echo "  ${RUN_FINALISTS[$index]} seed=${RUN_SEEDS[$index]}"
done

if [[ "${DRY_RUN}" == true ]]; then
  echo "Dry run: no validation preparation or Slurm submission was performed."
  for index in "${!RUN_FINALISTS[@]}"; do
    echo "  ${SBATCH_BIN} --parsable scripts/slurm_train_cawfe_latte_finalist_a10080.sh ${RUN_FINALISTS[$index]} ${RUN_SEEDS[$index]}"
  done
  exit 0
fi

# Build or reuse the one dataset-identified, fire/no-fire-stratified subset
# before allocating GPUs. Every job validates and reuses this same artifact.
"${PYTHON_BIN}" scripts/prepare_cawfe_latte_screening_validation.py

for index in "${!RUN_FINALISTS[@]}"; do
  finalist="${RUN_FINALISTS[$index]}"
  seed="${RUN_SEEDS[$index]}"
  JOB_ID="$("${SBATCH_BIN}" --parsable scripts/slurm_train_cawfe_latte_finalist_a10080.sh "${finalist}" "${seed}")"
  echo "${finalist} seed=${seed} -> job ${JOB_ID}"
done
