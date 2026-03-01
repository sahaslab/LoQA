#!/bin/bash
echo "=========================================="
echo "        Evaluation Script"
echo "=========================================="

# Configuring the path and stuffs
REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
source "$REPO_ROOT/.env"

# Change to repo root to ensure relative paths work
cd "$REPO_ROOT"

# parameter for evaluation script
SPLIT_NAME="gold-test" #"train" #"test" #"dev"
QG_PROMPT_VERSION="zs-v0"
PD_PATH="$REPO_ROOT/Outputs/pd/"
PD_PROMPT_VERSION="zs-v0"
RM_THRESHOLD=0.85
VERBOSE_EVAL=true
EVALUATION_PATH="$REPO_ROOT/Outputs/ev/"
DO_COMPLEX_MATCH=true
NUM_SAMPLES=-1

# paraters for the scoring script
OVERALL_SCORES=true
ROLE_WISE_SCORES=false # make true when you want role-wise scores
SAVE_SCORES=true
USE_WANDB=false
WANDB_PROJECT="loqa-scores"
VERBOSE_SCORE=true
SCORES_PATH="$REPO_ROOT/Outputs/sc/"

# Log directory
LOG_DIR="$REPO_ROOT/Outputs/logs"
mkdir -p "$LOG_DIR"

# Add date and time to log file names for uniqueness
CURRENT_DATETIME=$(date +"%Y-%m-%d")
LOG_DIR_WITH_DATE="$LOG_DIR/$CURRENT_DATETIME"
mkdir -p "$LOG_DIR_WITH_DATE"
LOG_DIR="$LOG_DIR_WITH_DATE"

# Evaluate both prediction types
run_evaluation() {
    local QG_MODEL_NAME=$1
    local PD_MODEL_NAME=$2
    local DATASET_NAME=$3
    local QUESTION_TYPE=$4

    local LOG_FILE="$LOG_DIR/eval-${QUESTION_TYPE}-${QG_MODEL_NAME}-${DATASET_NAME}-${SPLIT_NAME}-${PD_MODEL_NAME}.out"

    echo "Doing evaluation for Dataset: $DATASET_NAME, Split: $SPLIT_NAME, PD Model: $PD_MODEL_NAME, PD Prompt Version: $PD_PROMPT_VERSION"
    echo "Logging to: $LOG_FILE"
    echo ""
    
    # Build arguments
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
    [[ "$VERBOSE_EVAL" == true ]] && ARGS+=(--verbose)

    time python Code/scripts/4_evaluation.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
}

run_scoring() {
    local QG_MODEL_NAME=$1
    local PD_MODEL_NAME=$2
    local DATASET_NAME=$3
    local QUESTION_TYPE=$4

    local LOG_FILE="$LOG_DIR/score-${QUESTION_TYPE}-${QG_MODEL_NAME}-${DATASET_NAME}-${SPLIT_NAME}-${PD_MODEL_NAME}.out"

    echo "Doing scoring for QG Model: $QG_MODEL_NAME, PD Model: $PD_MODEL_NAME, Dataset: $DATASET_NAME, Question Type: $QUESTION_TYPE"
    echo "Logging to: $LOG_FILE"
    echo ""

    ARGS=(
        --dataset-name "$DATASET_NAME"
        --split-name "$SPLIT_NAME"
        --pred-key "$QUESTION_TYPE"
        --qg-model-name "$QG_MODEL_NAME"
        --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-model-name "$PD_MODEL_NAME"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --rm-threshold "$RM_THRESHOLD"
        --wandb-project "$WANDB_PROJECT"
        --evaluation-path "$EVALUATION_PATH"
        --scores-path "$SCORES_PATH"
    )
    [[ "$DO_COMPLEX_MATCH" == true ]] && ARGS+=(--do-complex-match)
    [[ "$OVERALL_SCORES" == true ]] && ARGS+=(--overall-scores)
    [[ "$ROLE_WISE_SCORES" == true ]] && ARGS+=(--role-wise-scores)
    [[ "$SAVE_SCORES" == true ]] && ARGS+=(--save-scores)
    [[ "$USE_WANDB" == true ]] && ARGS+=(--use-wandb)
    [[ "$VERBOSE_SCORE" == true ]] && ARGS+=(--verbose)

    time python Code/scripts/5_scoring.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
}

# Run evaluation for each dataset and prediction key

# #for LoQA models
QG_MODEL_NAMES=("gpt-oss-120b")
PD_MODEL_NAMES=("gpt-oss-120b")
DATASETS=("DiscourseEE") #"CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT"
QUESTION_TYPES=("optimized_loqa") #"schema" "cot-schema" "loqa" "optimized_loqa"

for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
    for PD_MODEL_NAME in "${PD_MODEL_NAMES[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
                echo "QG Model: $QG_MODEL_NAME, PD Model: $PD_MODEL_NAME"
                echo "Evaluating with dataset: $DATASET, Question Type: $QUESTION_TYPE"
                run_evaluation "$QG_MODEL_NAME" "$PD_MODEL_NAME" "$DATASET" "$QUESTION_TYPE"
                run_scoring "$QG_MODEL_NAME" "$PD_MODEL_NAME" "$DATASET" "$QUESTION_TYPE"
            done
        done
    done
done

#for Schema models (QG and PD models are same)
# QG_MODEL_NAMES=("qwen3-4b" "qwen3-8b") # 'gpt-5-mini' 'gpt-4.1-mini' 'gpt-oss-120b' 'qwen3-32b'
# DATASETS=("CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT")
# QUESTION_TYPES=("schema" "cot-schema")

# for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
#     for DATASET in "${DATASETS[@]}"; do
#         for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
#             echo "QG Model: $QG_MODEL_NAME, PD Model: $QG_MODEL_NAME"
#             echo "Evaluating with dataset: $DATASET, Question Type: $QUESTION_TYPE"
#             run_evaluation "$QG_MODEL_NAME" "$QG_MODEL_NAME" "$DATASET" "$QUESTION_TYPE"
#             run_scoring "$QG_MODEL_NAME" "$QG_MODEL_NAME" "$DATASET" "$QUESTION_TYPE"
#         done
#     done
# done