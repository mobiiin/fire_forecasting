#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN=false
EXACT_RUNS=()
BASELINES=()
SEEDS=()

usage() {
  echo "Usage:"
  echo "  bash $0"
  echo "  bash $0 --run NAME[:SEED] [--run NAME[:SEED] ...] [--dry-run]"
  echo "  bash $0 [--baseline NAME ...] [--seed SEED ...] [--dry-run]"
  echo ""
  echo "Default: 9 learned seed jobs plus one Persistence and one Linear job (11 total)."
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "$1 requires a value" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      require_value "$@"
      EXACT_RUNS+=("$2")
      shift 2
      ;;
    --baseline)
      require_value "$@"
      BASELINES+=("$2")
      shift 2
      ;;
    --seed)
      require_value "$@"
      SEEDS+=("$2")
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

if [[ ${#EXACT_RUNS[@]} -gt 0 && ( ${#BASELINES[@]} -gt 0 || ${#SEEDS[@]} -gt 0 ) ]]; then
  echo "--run cannot be combined with --baseline or --seed" >&2
  exit 2
fi

mapfile -t REGISTERED_BASELINES < <("${PYTHON_BIN}" scripts/run_table2_baseline.py --list-baselines)
mapfile -t REGISTERED_SEEDS < <("${PYTHON_BIN}" scripts/run_table2_baseline.py --list-seeds)
RUN_BASELINES=()
RUN_SEEDS=()
RUN_KEYS=()

is_deterministic() {
  [[ "$1" == "persistence" || "$1" == "linear_extrapolation" ]]
}

add_run() {
  local baseline="$1"
  local seed="$2"
  local key="${baseline}:${seed}"
  local existing
  for existing in "${RUN_KEYS[@]}"; do
    if [[ "${existing}" == "${key}" ]]; then
      return
    fi
  done
  RUN_KEYS+=("${key}")
  RUN_BASELINES+=("${baseline}")
  RUN_SEEDS+=("${seed}")
}

if [[ ${#EXACT_RUNS[@]} -gt 0 ]]; then
  for spec in "${EXACT_RUNS[@]}"; do
    baseline="${spec%%:*}"
    if [[ "${spec}" == *:* ]]; then
      seed="${spec#*:}"
    else
      seed=""
    fi
    if [[ -z "${baseline}" || "${seed}" == *:* ]]; then
      echo "Invalid --run ${spec}; use deterministic_name or learned_name:seed" >&2
      exit 2
    fi
    if is_deterministic "${baseline}"; then
      if [[ -n "${seed}" ]]; then
        echo "Deterministic baseline ${baseline} must not include a seed" >&2
        exit 2
      fi
    elif [[ -z "${seed}" ]]; then
      echo "Learned baseline ${baseline} requires NAME:SEED" >&2
      exit 2
    fi
    add_run "${baseline}" "${seed}"
  done
else
  if [[ ${#BASELINES[@]} -eq 0 ]]; then
    BASELINES=("${REGISTERED_BASELINES[@]}")
  fi
  if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=("${REGISTERED_SEEDS[@]}")
  fi
  for baseline in "${BASELINES[@]}"; do
    if is_deterministic "${baseline}"; then
      add_run "${baseline}" ""
    else
      for seed in "${SEEDS[@]}"; do
        add_run "${baseline}" "${seed}"
      done
    fi
  done
fi

for index in "${!RUN_BASELINES[@]}"; do
  baseline="${RUN_BASELINES[$index]}"
  seed="${RUN_SEEDS[$index]}"
  baseline_registered=false
  for registered in "${REGISTERED_BASELINES[@]}"; do
    [[ "${baseline}" == "${registered}" ]] && baseline_registered=true
  done
  if [[ "${baseline_registered}" != true ]]; then
    echo "Unknown Table 2 baseline: ${baseline}" >&2
    exit 2
  fi
  if [[ -n "${seed}" ]]; then
    seed_registered=false
    for registered in "${REGISTERED_SEEDS[@]}"; do
      [[ "${seed}" == "${registered}" ]] && seed_registered=true
    done
    if [[ "${seed_registered}" != true ]]; then
      echo "Unregistered Table 2 seed: ${seed}" >&2
      exit 2
    fi
  fi
done

"${PYTHON_BIN}" scripts/check_table2_baseline_configs.py

echo "Selected Table 2 jobs (${#RUN_BASELINES[@]}):"
for index in "${!RUN_BASELINES[@]}"; do
  echo "  ${RUN_BASELINES[$index]} seed=${RUN_SEEDS[$index]:-deterministic}"
done

if [[ "${DRY_RUN}" == true ]]; then
  echo "Dry run: no validation preparation or Slurm submission was performed."
  for index in "${!RUN_BASELINES[@]}"; do
    echo "  ${SBATCH_BIN} --parsable scripts/slurm_train_table2_baseline_a10080.sh ${RUN_BASELINES[$index]} ${RUN_SEEDS[$index]}"
  done
  exit 0
fi

HAS_LEARNED=false
for baseline in "${RUN_BASELINES[@]}"; do
  if ! is_deterministic "${baseline}"; then
    HAS_LEARNED=true
    break
  fi
done
if [[ "${HAS_LEARNED}" == true ]]; then
  "${PYTHON_BIN}" scripts/prepare_cawfe_latte_screening_validation.py
fi

for index in "${!RUN_BASELINES[@]}"; do
  baseline="${RUN_BASELINES[$index]}"
  seed="${RUN_SEEDS[$index]}"
  if [[ -n "${seed}" ]]; then
    JOB_ID="$("${SBATCH_BIN}" --parsable scripts/slurm_train_table2_baseline_a10080.sh "${baseline}" "${seed}")"
  else
    JOB_ID="$("${SBATCH_BIN}" --parsable scripts/slurm_train_table2_baseline_a10080.sh "${baseline}")"
  fi
  echo "${baseline} seed=${seed:-deterministic} -> job ${JOB_ID}"
done
