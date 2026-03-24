#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        3. Prediction"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.env"

# ── Configuration ────────────────────────────────────────────
DATASETS=("CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT")
QUESTION_TYPES=("loqa")                 # loqa | dynamicQ | optimized_loqa | schema | cot-schema
SPLITS=("test")                         # train | dev | test
NUM_SAMPLES=-1                          # -1 = all
USE_ASYNC="1"
BATCH_SIZE=100

# QG model (must match what was used in step 2)
QG_MODEL_NAMES=("gpt-oss-120b")
QG_PROMPT_VERSION="zs-v0"

# PD model (index-aligned arrays)
PD_MODEL_NAMES=("gpt-oss-120b")
PD_MODEL_ORIGINS=("dartmouth")
PD_MODEL_ACCESS_STRINGS=("openai.gpt-oss-120b")
PD_TEMPERATURE=0.0
PD_GPU_UTIL=0.9
PD_PROMPT_VERSION="zs-v0"
REASONING_EFFORT="none"

# Paths
QG_OUTPUT_PATH="$REPO_ROOT/Outputs/qg"
PD_OUTPUT_PATH="$REPO_ROOT/Outputs/pd"
PROMPT_DIR="$REPO_ROOT/Prompts"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache}"
# ─────────────────────────────────────────────────────────────

run_prediction() {
    local QG_MODEL_NAME=$1 PD_MODEL_NAME=$2 PD_MODEL_ACCESS_STRING=$3
    local DATASET_NAME=$4 QUESTION_TYPE=$5 SPLIT_NAME=$6 PD_MODEL_ORIGIN=$7

    echo "QG: $QG_MODEL_NAME | PD: $PD_MODEL_NAME | Dataset: $DATASET_NAME | Type: $QUESTION_TYPE | Split: $SPLIT_NAME"

    ARGS=(
        --dataset-name "$DATASET_NAME"
        --split-name "$SPLIT_NAME"
        --question-type "$QUESTION_TYPE"
        --qg-model-name "$QG_MODEL_NAME"
        --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-model-origin "$PD_MODEL_ORIGIN"
        --pd-model-name "$PD_MODEL_NAME"
        --pd-model-access-string "$PD_MODEL_ACCESS_STRING"
        --pd-temperature "$PD_TEMPERATURE"
        --pd-gpu-util "$PD_GPU_UTIL"
        --cache-dir "$CACHE_DIR"
        --reasoning-effort "$REASONING_EFFORT"
        --num-samples "$NUM_SAMPLES"
        --qg-output-path "$QG_OUTPUT_PATH"
        --pd-output-path "$PD_OUTPUT_PATH"
        --batch-size "$BATCH_SIZE"
        --hf-token "${HF_TOKEN:-}"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --prompt-dir "$PROMPT_DIR"
    )
    [[ "$USE_ASYNC" == "1" ]] && ARGS+=(--use-async)

    time python Code/scripts/3_predictions.py "${ARGS[@]}"
}

cd "$REPO_ROOT"
for SPLIT in "${SPLITS[@]}"; do
    for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
        for i in "${!PD_MODEL_NAMES[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
                    run_prediction "$QG_MODEL_NAME" \
                        "${PD_MODEL_NAMES[$i]}" "${PD_MODEL_ACCESS_STRINGS[$i]}" \
                        "$DATASET" "$QUESTION_TYPE" "$SPLIT" "${PD_MODEL_ORIGINS[$i]}"
                done
            done
        done
    done
done
