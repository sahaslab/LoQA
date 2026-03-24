#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        2. Question Generation"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.env"

# ── Configuration ────────────────────────────────────────────
DATASETS=("CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT")
QUESTION_TYPES=("loqa")                 # loqa | dynamicQ | schema | cot-schema
SPLIT_NAME="test"                       # train | dev | test
NUM_SAMPLES=-1                          # -1 = all valid samples
USE_ASYNC="1"                           # "1" to enable async
BATCH_SIZE=100

# QG model
QG_MODEL_ORIGIN="dartmouth"             # dartmouth | openai | google | vllm-local | vllm-serve
QG_MODEL_NAMES=("gpt-oss-120b")
QG_MODEL_ACCESS_STRING="openai.gpt-oss-120b"
QG_PROMPT_VERSION="zs-v0"
QG_TEMPERATURE=0.0
QG_GPU_UTIL=0.9
REASONING_EFFORT="none"

# Paths
DATASET_ROOT="$REPO_ROOT/Dataset"
OUTPUT_PATH="$REPO_ROOT/Outputs/qg"
PROMPT_DIR="$REPO_ROOT/Prompts"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache}"
# ─────────────────────────────────────────────────────────────

run_question_generation() {
    local DATASET_NAME=$1
    local QUESTION_TYPE=$2
    local QG_MODEL_NAME=$3

    ARGS=(
        --dataset-root "$DATASET_ROOT"
        --dataset-name "$DATASET_NAME"
        --split-name "$SPLIT_NAME"
        --question-type "$QUESTION_TYPE"
        --qg-model-name "$QG_MODEL_NAME"
        --qg-model-origin "$QG_MODEL_ORIGIN"
        --qg-model-access-string "$QG_MODEL_ACCESS_STRING"
        --qg-temperature "$QG_TEMPERATURE"
        --qg-gpu-util "$QG_GPU_UTIL"
        --cache-dir "$CACHE_DIR"
        --reasoning-effort "$REASONING_EFFORT"
        --num-samples "$NUM_SAMPLES"
        --batch-size "$BATCH_SIZE"
        --qg-prompt-version "$QG_PROMPT_VERSION"
        --output-path "$OUTPUT_PATH"
        --prompt-dir "$PROMPT_DIR"
        --hf-token "${HF_TOKEN:-}"
    )
    [[ "$USE_ASYNC" == "1" ]] && ARGS+=(--use-async)

    echo "Question Type: $QUESTION_TYPE | Dataset: $DATASET_NAME | Split: $SPLIT_NAME | Model: $QG_MODEL_NAME"
    time python Code/scripts/2_loq_greneration.py "${ARGS[@]}"
}

cd "$REPO_ROOT"
for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
            run_question_generation "$DATASET" "$QUESTION_TYPE" "$QG_MODEL_NAME"
        done
    done
done
