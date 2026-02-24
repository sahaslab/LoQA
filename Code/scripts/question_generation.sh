#!/bin/bash
echo "=========================================="
echo "         Question Generation Script"
echo "=========================================="

# Configuring the path and stuffs
REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
source "$REPO_ROOT/.env"

# Common Variables - Modify these as needed
SPLIT_NAME="gold-test" #"train" #"test" #"dev"
QG_MODEL_NAME="gpt-oss-120b"
QG_MODEL_ORIGIN="dartmouth"  #"dartmouth" #"dartmouth" 
QG_MODEL_ACCESS_STRING='openai.gpt-oss-120b' #"openai.gpt-4.1-mini-2025-04-14" #"openai.gpt-oss-120b" #"openai.gpt-4.1-mini-2025-04-14" #"openai.gpt-oss-120b"
REASONING_EFFORT="high"
NUM_SAMPLES=-1 #if set to -1, all valid samples will be processed
USE_ASYNC="1"  # set to empty string to disable async: USE_ASYNC="d"
BATCH_SIZE=200
QG_PROMPT_VERSION="zs-v0"

#variable that usually would not change every on every run
DATASET_ROOT="$REPO_ROOT/Dataset"
QG_TEMPERATURE=0.0
QG_GPU_UTIL=0.9
CACHE_DIR="/dartfs-hpc/rc/home/j/f006f3j/lab/shared"
OUTPUT_PATH="$REPO_ROOT/Outputs/qg/" #change directory name to raw for testing purposes
PROMPT_DIR="$REPO_ROOT/Prompts"

# Build arguments
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
        --hf-token "$HF_TOKEN"
    )
    [ "$USE_ASYNC" = "1" ] && ARGS+=(--use-async)

    # Run
    cd "$REPO_ROOT"
    echo "Current directory: $(pwd)"
    echo "Question Type: $QUESTION_TYPE, Dataset: $DATASET_NAME" "Split: $SPLIT_NAME" "Model: $QG_MODEL_NAME" "Num Samples: $NUM_SAMPLES"
    echo ""
    time python Code/scripts/2_loq_greneration.py "${ARGS[@]}"
}

DATASETS=("CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT") #"CaseReportBench" "PHEE" "DiscourseEE"
PRED_KEYS=("schema" "cot-schema") #"loqa" "schema"
QG_MODEL_NAMES=("gpt-5-mini" "gpt-4.1-mini" "qwen3-4b" "qwen3-8b" "qwen3-32b")
for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for PRED_KEY in "${PRED_KEYS[@]}"; do
            echo "Generating questions for dataset: $DATASET, question-type: $PRED_KEY, QG Model: $QG_MODEL_NAME"
            run_question_generation "$DATASET" "$PRED_KEY" "$QG_MODEL_NAME"
        done
    done
done
