#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        8. Generate with Fine-tuned QG"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.env"

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "ERROR: TINKER_API_KEY not set."; exit 1; }

# ── Configuration ────────────────────────────────────────────
DATASETS=("CaseReportBench" "DiscourseEE" "MACCROBAT" "PHEE")
SPLITS=("dev" "test")

QG_MODEL_NAMES=("qwen3-8b-mix-balanced-v1")
RUN_DIR="$REPO_ROOT/Outputs/ft_runs/qwen3-8b-mix-balanced-v1"
OUTPUT_PATH="$REPO_ROOT/Outputs/ft_outputs/qg"

TEMPERATURE="0.7"
TOP_P="0.8"
MAX_NEW_TOKENS="512"
N_SAMPLES="-1"                          # -1 = all rows

LOG_DIR="$REPO_ROOT/Outputs/logs/$(date +%Y-%m-%d)"
mkdir -p "$LOG_DIR"
# ─────────────────────────────────────────────────────────────

run_generation() {
    local DATASET_NAME=$1 SPLIT_NAME=$2 QG_MODEL_NAME=$3

    ARGS=(
        --dataset-root      "$REPO_ROOT/Dataset"
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
    [[ "$N_SAMPLES" != "-1" ]] && ARGS+=(--n-samples "$N_SAMPLES")

    echo "[QG-FT] Dataset: $DATASET_NAME | Split: $SPLIT_NAME | Model: $QG_MODEL_NAME"
    python Code/scripts/9_tinker_qg_generate_simple.py "${ARGS[@]}"
}

cd "$REPO_ROOT"
for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            LOG_FILE="$LOG_DIR/ft-qg-${QG_MODEL_NAME}-${DATASET}-${SPLIT}.out"
            run_generation "$DATASET" "$SPLIT" "$QG_MODEL_NAME" 2>&1 | tee "$LOG_FILE"
        done
    done
done
