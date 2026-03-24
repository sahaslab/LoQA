#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        4. Evaluation & Scoring"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.env"
cd "$REPO_ROOT"

# ── Configuration ────────────────────────────────────────────
DATASETS=("CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT")
QUESTION_TYPES=("loqa")                 # loqa | dynamicQ | optimized_loqa | schema | cot-schema
SPLITS=("test")

# Must match models used in steps 2-3
QG_MODEL_NAMES=("gpt-oss-120b")
PD_MODEL_NAMES=("gpt-oss-120b")
QG_PROMPT_VERSION="zs-v0"
PD_PROMPT_VERSION="zs-v0"

# Evaluation parameters
RM_THRESHOLD=0.85
DO_COMPLEX_MATCH=true
VERBOSE_EVAL=true
NUM_SAMPLES=-1

# Scoring parameters
OVERALL_SCORES=true
ROLE_WISE_SCORES=false
SAVE_SCORES=true
VERBOSE_SCORE=true

# Paths
PD_PATH="$REPO_ROOT/Outputs/pd"
EVALUATION_PATH="$REPO_ROOT/Outputs/ev"
SCORES_PATH="$REPO_ROOT/Outputs/sc"

LOG_DIR="$REPO_ROOT/Outputs/logs/$(date +%Y-%m-%d)"
mkdir -p "$LOG_DIR"
# ─────────────────────────────────────────────────────────────

run_evaluation() {
    local QG_MODEL_NAME=$1 PD_MODEL_NAME=$2 DATASET_NAME=$3
    local QUESTION_TYPE=$4 SPLIT_NAME=$5
    local LOG_FILE="$LOG_DIR/eval-${QUESTION_TYPE}-${QG_MODEL_NAME}-${DATASET_NAME}-${SPLIT_NAME}-${PD_MODEL_NAME}.out"

    echo "[EVAL] Dataset: $DATASET_NAME | Type: $QUESTION_TYPE | QG: $QG_MODEL_NAME | PD: $PD_MODEL_NAME"

    ARGS=(
        --dataset-name "$DATASET_NAME"
        --split-name "$SPLIT_NAME"
        --question-type "$QUESTION_TYPE"
        --qg-model-name "$QG_MODEL_NAME"
        --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-path "$PD_PATH"
        --pd-model-name "$PD_MODEL_NAME"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --rm-threshold "$RM_THRESHOLD"
        --evaluation-path "$EVALUATION_PATH"
        --num-samples "$NUM_SAMPLES"
    )
    [[ "$DO_COMPLEX_MATCH" == true ]] && ARGS+=(--do-complex-match)
    [[ "$VERBOSE_EVAL"     == true ]] && ARGS+=(--verbose)

    time python Code/scripts/4_evaluation.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
}

run_scoring() {
    local QG_MODEL_NAME=$1 PD_MODEL_NAME=$2 DATASET_NAME=$3
    local QUESTION_TYPE=$4 SPLIT_NAME=$5
    local LOG_FILE="$LOG_DIR/score-${QUESTION_TYPE}-${QG_MODEL_NAME}-${DATASET_NAME}-${SPLIT_NAME}-${PD_MODEL_NAME}.out"

    echo "[SCORE] Dataset: $DATASET_NAME | Type: $QUESTION_TYPE | QG: $QG_MODEL_NAME | PD: $PD_MODEL_NAME"

    ARGS=(
        --dataset-name "$DATASET_NAME"
        --split-name "$SPLIT_NAME"
        --pred-key "$QUESTION_TYPE"
        --qg-model-name "$QG_MODEL_NAME"
        --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-model-name "$PD_MODEL_NAME"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --rm-threshold "$RM_THRESHOLD"
        --evaluation-path "$EVALUATION_PATH"
        --scores-path "$SCORES_PATH"
    )
    [[ "$DO_COMPLEX_MATCH"  == true ]] && ARGS+=(--do-complex-match)
    [[ "$OVERALL_SCORES"    == true ]] && ARGS+=(--overall-scores)
    [[ "$ROLE_WISE_SCORES"  == true ]] && ARGS+=(--role-wise-scores)
    [[ "$SAVE_SCORES"       == true ]] && ARGS+=(--save-scores)
    [[ "$VERBOSE_SCORE"     == true ]] && ARGS+=(--verbose)

    time python Code/scripts/5_scoring.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
}

for SPLIT in "${SPLITS[@]}"; do
    for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
        for PD_MODEL_NAME in "${PD_MODEL_NAMES[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
                    run_evaluation "$QG_MODEL_NAME" "$PD_MODEL_NAME" "$DATASET" "$QUESTION_TYPE" "$SPLIT"
                    run_scoring    "$QG_MODEL_NAME" "$PD_MODEL_NAME" "$DATASET" "$QUESTION_TYPE" "$SPLIT"
                done
            done
        done
    done
done
