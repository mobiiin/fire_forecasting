#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_data_preparation_pipeline.sh CONFIG [OPTIONS]

Options:
  --derived-config PATH       Derived threshold config path
  --percentile FLOAT          Threshold percentile (default: 1.0)
  --pattern PATTERN           Normalization/sample pattern (default: consecutive5_h10)
  --all-patterns              Build all enabled temporal patterns (default)
  --skip-engineered           Skip engineered full-frame build
  --skip-thresholds           Skip threshold estimation
  --skip-targets              Skip target construction
  --skip-patches              Skip patch-index construction
  --skip-samples              Skip temporal sample-index construction
  --skip-normalization        Skip normalization
  --skip-inspection           Skip inspection
  --make-quicklooks           Save debug visualizations
  --overwrite                 Pass overwrite where supported
  --max-frames-per-fire N     Debug-only frame limit
  --dry-run                   Print commands without running them
  --help                      Show this help
EOF
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    [[ $# -ge 1 ]] && [[ "$1" == "--help" || "$1" == "-h" ]] && exit 0
    exit 1
fi

CONFIG="$1"
shift
if [[ ! -f "$CONFIG" ]]; then
    echo "Config file does not exist: $CONFIG" >&2
    exit 1
fi
CONFIG="$(realpath "$CONFIG")"
CONFIG_STEM="$(basename "$CONFIG")"
CONFIG_STEM="${CONFIG_STEM%.*}"

DERIVED_CONFIG=""
PERCENTILE="5.0"
PATTERN="single1_h10"
ALL_PATTERNS=1
MAKE_QUICKLOOKS=0
DRY_RUN=0
OVERWRITE=0
SKIP_ENGINEERED=0
SKIP_THRESHOLDS=0
SKIP_TARGETS=0
SKIP_PATCHES=0
SKIP_SAMPLES=0
SKIP_NORMALIZATION=0
SKIP_INSPECTION=0
MAX_FRAMES_PER_FIRE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --derived-config) DERIVED_CONFIG="$2"; shift 2 ;;
        --percentile) PERCENTILE="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --all-patterns) ALL_PATTERNS=1; shift ;;
        --skip-engineered) SKIP_ENGINEERED=1; shift ;;
        --skip-thresholds) SKIP_THRESHOLDS=1; shift ;;
        --skip-targets) SKIP_TARGETS=1; shift ;;
        --skip-patches) SKIP_PATCHES=1; shift ;;
        --skip-samples) SKIP_SAMPLES=1; shift ;;
        --skip-normalization) SKIP_NORMALIZATION=1; shift ;;
        --skip-inspection) SKIP_INSPECTION=1; shift ;;
        --make-quicklooks) MAKE_QUICKLOOKS=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --max-frames-per-fire) MAX_FRAMES_PER_FIRE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$DERIVED_CONFIG" ]]; then
    DERIVED_CONFIG="configs/derived/${CONFIG_STEM}_with_fire_mask_thresholds.yaml"
fi
if [[ ! "$DERIVED_CONFIG" = /* ]]; then
    DERIVED_CONFIG="$(realpath -m "$DERIVED_CONFIG")"
fi

for required in \
    scripts/build_engineered_frame_dataset.py \
    scripts/estimate_fire_mask_thresholds.py \
    scripts/build_target_dataset.py \
    scripts/build_patch_index.py \
    scripts/build_temporal_sample_index.py \
    scripts/compute_processed_dataset_normalization.py \
    scripts/inspect_processed_dataset.py; do
    [[ -f "$required" ]] || { echo "Required script is missing: $required" >&2; exit 1; }
done

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="artifacts/logs/data_preparation/${CONFIG_STEM}_${TIMESTAMP}"
mkdir -p "$LOG_DIR" configs/derived
touch "$LOG_DIR/commands.txt"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

write_status() {
    local status="$1"
    local finished_at
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cat > "$LOG_DIR/status.json" <<EOF
{
  "status": "$status",
  "config": "$(printf '%s' "$CONFIG" | sed 's/"/\\"/g')",
  "derived_config": "$(printf '%s' "$DERIVED_CONFIG" | sed 's/"/\\"/g')",
  "pattern": "$PATTERN",
  "percentile": $PERCENTILE,
  "started_at": "$STARTED_AT",
  "finished_at": "$finished_at",
  "log_dir": "$(printf '%s' "$LOG_DIR" | sed 's/"/\\"/g')"
}
EOF
}
on_error() { write_status "failed"; }
trap on_error ERR

{
    echo "hostname: $(hostname)"
    echo "date: $(date -u)"
    echo "pwd: $(pwd)"
    echo "python_version: $(python --version 2>&1)"
    echo "python_path: $(command -v python || true)"
    echo "git_head: $(git rev-parse HEAD 2>/dev/null || true)"
    echo "config: $CONFIG"
    echo "derived_config: $DERIVED_CONFIG"
    echo "percentile: $PERCENTILE"
    echo "pattern: $PATTERN"
} > "$LOG_DIR/environment.txt"

run_cmd() {
    echo
    echo "============================================================"
    printf 'Running:'; printf ' %q' "$@"; echo
    echo "============================================================"
    printf '%q ' "$@" >> "$LOG_DIR/commands.txt"
    echo >> "$LOG_DIR/commands.txt"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    "$@"
}

if [[ -n "$MAX_FRAMES_PER_FIRE" ]]; then
    echo "DEBUG MODE: max frames per fire is set. Dataset is incomplete."
fi

if [[ "$SKIP_ENGINEERED" == "0" ]]; then
    cmd=(python scripts/build_engineered_frame_dataset.py --config "$CONFIG")
    [[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)
    [[ -n "$MAX_FRAMES_PER_FIRE" ]] && cmd+=(--max_frames_per_fire "$MAX_FRAMES_PER_FIRE")
    run_cmd "${cmd[@]}"
fi

if [[ "$SKIP_THRESHOLDS" == "0" ]]; then
    cmd=(python scripts/estimate_fire_mask_thresholds.py --config "$CONFIG" --percentile "$PERCENTILE" --update_config --derived_config_path "$DERIVED_CONFIG")
    [[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)
    run_cmd "${cmd[@]}"
else
    if [[ "$DRY_RUN" == "0" && ! -f "$DERIVED_CONFIG" ]]; then
        if ! python - "$CONFIG" <<'PY'
import sys
from src.config import load_config
from src.data.fire_mask_thresholds import resolve_frozen_thresholds
try:
    resolve_frozen_thresholds(load_config(sys.argv[1]), sys.argv[1], require=True)
except Exception:
    raise SystemExit(1)
PY
        then
            if [[ "$SKIP_TARGETS" == "0" ]]; then
                echo "Cannot build targets without thresholds. Run threshold estimation or provide --derived-config." >&2
                exit 1
            fi
            echo "WARNING: thresholds skipped and no frozen threshold config was found."
        fi
    fi
fi

TARGET_CONFIG="$DERIVED_CONFIG"
if [[ "$SKIP_THRESHOLDS" == "1" && ! -f "$TARGET_CONFIG" ]]; then TARGET_CONFIG="$CONFIG"; fi

if [[ "$SKIP_TARGETS" == "0" ]]; then
    cmd=(python scripts/build_target_dataset.py --config "$TARGET_CONFIG")
    [[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)
    run_cmd "${cmd[@]}"
fi
if [[ "$SKIP_PATCHES" == "0" ]]; then
    cmd=(python scripts/build_patch_index.py --config "$TARGET_CONFIG")
    [[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)
    run_cmd "${cmd[@]}"
fi
if [[ "$SKIP_SAMPLES" == "0" ]]; then
    if [[ "$ALL_PATTERNS" == "1" ]]; then
        run_cmd python scripts/build_temporal_sample_index.py --config "$TARGET_CONFIG" --pattern all
    else
        run_cmd python scripts/build_temporal_sample_index.py --config "$TARGET_CONFIG" --pattern "$PATTERN"
    fi
fi
if [[ "$SKIP_NORMALIZATION" == "0" ]]; then
    run_cmd python scripts/compute_processed_dataset_normalization.py --config "$TARGET_CONFIG" --pattern "$PATTERN"
fi
if [[ "$SKIP_INSPECTION" == "0" ]]; then
    run_cmd python scripts/inspect_processed_dataset.py --config "$TARGET_CONFIG"
fi

if [[ "$MAKE_QUICKLOOKS" == "1" ]]; then
    run_cmd python scripts/visualize_engineered_frames.py --config "$TARGET_CONFIG" --split train --mode save --output_dir artifacts/data_debug/engineered_frames
    run_cmd python scripts/visualize_patch_index.py --config "$TARGET_CONFIG" --split train --mode save --output_dir artifacts/data_debug/patch_index
    run_cmd python scripts/visualize_targets.py --config "$TARGET_CONFIG" --split train --mode save --output_dir artifacts/data_debug/targets
    run_cmd python scripts/visualize_processed_samples.py --config "$TARGET_CONFIG" --pattern "$PATTERN" --split train --mode save --output_dir artifacts/data_debug/processed_samples
fi

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "success"
trap - ERR
echo
echo "Data preparation completed successfully."
echo "  Config used: $CONFIG"
echo "  Derived threshold config: $TARGET_CONFIG"
echo "  Logs: $LOG_DIR"
echo "  Next suggested inspection: python scripts/inspect_processed_dataset.py --config $TARGET_CONFIG"
echo "  Next suggested visualizations:"
echo "    python scripts/visualize_targets.py --config $TARGET_CONFIG --split train"
echo "    python scripts/visualize_processed_samples.py --config $TARGET_CONFIG --pattern $PATTERN --split train"
echo "  Next training command: python scripts/train_forecasting_model.py --config $TARGET_CONFIG"
echo "  Ablation examples:"
echo "    python scripts/train_forecasting_model.py --config configs/experiments/convlstm_consecutive5_h10.yaml"
echo "    python scripts/train_forecasting_model.py --config configs/experiments/convlstm_single1_h10.yaml"
echo "    python scripts/train_forecasting_model.py --config configs/experiments/convlstm_sparse5_h10.yaml"
