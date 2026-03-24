#!/bin/bash
echo "=========================================="
echo "        Scoring Script"
echo "=========================================="

# Configuring the path and stuffs
REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
source "$REPO_ROOT/.env"

# need to change these based on dataset and model
SPLIT_NAME="test"
QG_MODEL_NAME="gpt-5-mini"
QG_PROMPT_VERSION="zs-v0"
PD_MODEL_NAME="gpt-4.1-mini"
PD_PROMPT_VERSION="zs-v0"

# rarely have to change these
RM_THRESHOLD=0.85
OVERALL_SCORES=True
ROLE_WISE_SCORES=False #make only true when you want to see role-wise scores
SAVE_SCORES=True
USE_WANDB=False
WANDB_PROJECT="loqa-scores"
VERBOSE=True
EVALUATION_PATH="$REPO_ROOT/Outputs/ev/"

# Function to run scoring
run_scoring() {
    local DATASET_NAME=$1
    local PRED_KEY=$2

    echo "Doing scoring for Pred Key: $PRED_KEY, Dataset: $DATASET_NAME, PD Model: $PD_MODEL_NAME, PD Prompt Version: $PD_PROMPT_VERSION"
    echo ""

    ARGS=(
        --dataset-name "$DATASET_NAME"
        --split-name "$SPLIT_NAME"
        --pred-key "$PRED_KEY"
        --qg-model-name "$QG_MODEL_NAME"
        --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-model-name "$PD_MODEL_NAME"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --rm-threshold "$RM_THRESHOLD"
        --wandb-project "$WANDB_PROJECT"
        --evaluation-path "$EVALUATION_PATH"
    )
    [ "$OVERALL_SCORES" = "True" ] && ARGS+=(--overall-scores)
    [ "$ROLE_WISE_SCORES" = "True" ] && ARGS+=(--role-wise-scores)
    [ "$SAVE_SCORES" = "True" ] && ARGS+=(--save-scores)
    [ "$USE_WANDB" = "True" ] && ARGS+=(--use-wandb)
    [ "$VERBOSE" = "True" ] && ARGS+=(--verbose)

    python Code/scripts/5_scoring.py "${ARGS[@]}"
}

# Run scoring for each pred key
DATASETS=("CaseReportBench" "PHEE" "DiscourseEE")
PRED_KEYS=("loqa")
for DATASET in "${DATASETS[@]}"; do
    for PRED_KEY in "${PRED_KEYS[@]}"; do
        run_scoring "$DATASET" "$PRED_KEY"
    done
done

