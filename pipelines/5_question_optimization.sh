#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        5. Question Optimization"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.env"

# ── Configuration ────────────────────────────────────────────
DATASETS=("CaseReportBench")            # CaseReportBench | PHEE | DiscourseEE | MACCROBAT
SPLIT_NAME="train"
NUM_SAMPLES=-1                          # -1 = all valid samples

# QO model
QO_MODEL_NAME="gpt-oss-120b"
QO_MODEL_ORIGIN="dartmouth"
QO_MODEL_ACCESS_STRING="openai.gpt-oss-120b"
QO_TEMPERATURE=0.0
QO_GPU_UTIL=0.9
QO_REASONING_EFFORT="none"
QO_PROMPT_VERSION="zs-v0"
INITIAL_QO_PROMPT_VERSION="zs-v0"

# PD model (for argument extraction in the refinement loop)
PD_MODEL_NAME="gpt-oss-120b"
PD_MODEL_ORIGIN="dartmouth"
PD_MODEL_ACCESS_STRING="openai.gpt-oss-120b"
PD_TEMPERATURE=0.0
PD_GPU_UTIL=0.9
PD_REASONING_EFFORT="none"
PD_PROMPT_VERSION="zs-v0"

# Refinement loop
NUM_ITERATIONS=5
TARGET_SCORE=1.0
MAXIMUM_PATIENCE=3
VERBOSE="1"

# Paths
DATASET_ROOT="$REPO_ROOT/Dataset"
OUTPUT_PATH="$REPO_ROOT/Outputs/qo"
PROMPT_DIR="$REPO_ROOT/Prompts"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache}"

LOG_DIR="$REPO_ROOT/Outputs/logs/$(date +%Y-%m-%d)"
mkdir -p "$LOG_DIR"
# ─────────────────────────────────────────────────────────────

run_question_optimization() {
    local DATASET_NAME=$1

    ARGS=(
        --dataset-root "$DATASET_ROOT"
        --dataset-name "$DATASET_NAME"
        --split-name "$SPLIT_NAME"
        --num-samples "$NUM_SAMPLES"
        --qo-model-name "$QO_MODEL_NAME"
        --qo-model-origin "$QO_MODEL_ORIGIN"
        --qo-model-access-string "$QO_MODEL_ACCESS_STRING"
        --qo-temperature "$QO_TEMPERATURE"
        --qo-gpu-util "$QO_GPU_UTIL"
        --qo-reasoning-effort "$QO_REASONING_EFFORT"
        --qo-prompt-version "$QO_PROMPT_VERSION"
        --initial-qo-prompt-version "$INITIAL_QO_PROMPT_VERSION"
        --pd-model-name "$PD_MODEL_NAME"
        --pd-model-origin "$PD_MODEL_ORIGIN"
        --pd-model-access-string "$PD_MODEL_ACCESS_STRING"
        --pd-temperature "$PD_TEMPERATURE"
        --pd-gpu-util "$PD_GPU_UTIL"
        --pd-reasoning-effort "$PD_REASONING_EFFORT"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --num-iterations "$NUM_ITERATIONS"
        --target-score "$TARGET_SCORE"
        --maximum-patience "$MAXIMUM_PATIENCE"
        --output-path "$OUTPUT_PATH"
        --prompt-dir "$PROMPT_DIR"
        --cache-dir "$CACHE_DIR"
    )
    [[ "$VERBOSE" == "1" ]] && ARGS+=(--verbose)

    echo "Dataset: $DATASET_NAME | Split: $SPLIT_NAME | QO: $QO_MODEL_NAME | PD: $PD_MODEL_NAME"
    time python Code/scripts/6_q_optimization.py "${ARGS[@]}"
}

cd "$REPO_ROOT"
for DATASET in "${DATASETS[@]}"; do
    LOG_FILE="$LOG_DIR/qo-${QO_MODEL_NAME}-${DATASET}-${SPLIT_NAME}.out"
    echo "Log: $LOG_FILE"
    run_question_optimization "$DATASET" 2>&1 | tee "$LOG_FILE"
done
