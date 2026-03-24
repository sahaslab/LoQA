#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
SCRIPT_PATH="$REPO_ROOT/Code/scripts/9_tinker_qg_generate_simple.py"
source "$REPO_ROOT/.env"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is not set."
  exit 1
fi

QG_MODEL_NAMES=("qwen3-8b-mix-bln-v5")
RUN_DIR="$REPO_ROOT/Outputs/ft_runs/qwen3-8b-mix-balanced-v5"
DATASET_ROOT="$REPO_ROOT/Dataset"
OUTPUT_PATH="$REPO_ROOT/Outputs/ft_outputs/qg"

TEMPERATURE="0.7" # suggested for qwen
TOP_P="0.8"
MAX_NEW_TOKENS="512"
N_SAMPLES="-1"  # -1 means all rows

# Log directory
LOG_DIR="$REPO_ROOT/Outputs/logs"
mkdir -p "$LOG_DIR"

# Add date and time to log file names for uniqueness
CURRENT_DATETIME=$(date +"%Y-%m-%d")
LOG_DIR_WITH_DATE="$LOG_DIR/$CURRENT_DATETIME"
mkdir -p "$LOG_DIR_WITH_DATE"
LOG_DIR="$LOG_DIR_WITH_DATE"

cd "$REPO_ROOT"

run_tinker_qg_generation() {
    local DATASET_NAME=$1
    local SPLIT_NAME=$2
    local QG_MODEL_NAME=$3


    ARGS=(
      --dataset-root      "$DATASET_ROOT"
      --dataset-name      "$DATASET_NAME"
      --split-name        "$SPLIT_NAME"
      --question-type     "ft"
      --qg-model-name     "$QG_MODEL_NAME"
      --qg-prompt-version "zs-v0"
      --run-dir           "$RUN_DIR"
      --output-path       "$OUTPUT_PATH"
      --max-new-tokens    "$MAX_NEW_TOKENS"
      --temperature       "$TEMPERATURE"
      --top-p             "$TOP_P"
    )

    if [[ "$N_SAMPLES" != "-1" ]]; then
      ARGS+=(--n-samples "$N_SAMPLES")
    fi

    python "$SCRIPT_PATH" "${ARGS[@]}"
}


DATASETS=("CaseReportBench" "DiscourseEE" "MACCROBAT")
SPLITS=("dev" "test")

for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            LOG_FILE="$LOG_DIR/tinker-qg-${QG_MODEL_NAME}-${DATASET}-${SPLIT}.out"
            echo "[INFO] dataset=$DATASET  split=$SPLIT  model=$QG_MODEL_NAME"
            echo "Logging to: $LOG_FILE"
            echo ""
            run_tinker_qg_generation "$DATASET" "$SPLIT" "$QG_MODEL_NAME" 2>&1 | tee "$LOG_FILE"
        done
    done
done