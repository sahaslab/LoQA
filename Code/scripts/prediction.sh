#!/bin/bash

# Shell script to run predictions

echo "=========================================="
echo "Running predictions"
echo "=========================================="

# Common Variables - Modify these as needed
REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
source "$REPO_ROOT/.env"

SPLIT_NAME="gold-test" #"train" #"test" #"dev"
QG_PROMPT_VERSION="zs-v0" # Question generation prompt version
PD_MODEL_ORIGIN="dartmouth" #"dartmouth" 
REASONING_EFFORT="none"
NUM_SAMPLES=-1 #if set to -1, all samples will be processed
USE_ASYNC="1"  # set to empty string to disable async: USE_ASYNC=""
BATCH_SIZE=200
USE_WANDB="" # set to empty string to disable wandb: USE_WANDB="""
PRINT_PROMPT="" # set to "1" to print formatted prompt before each get_response: PRINT_PROMPT="1"

#variable that would not change every on every run
PD_TEMPERATURE=0.0
PD_GPU_UTIL=0.9
CACHE_DIR="/dartfs-hpc/rc/home/j/f006f3j/lab/shared"
QG_OUTPUT_PATH="$REPO_ROOT/Outputs/qg/"
PD_OUTPUT_PATH="$REPO_ROOT/Outputs/pd/" #change directory name to raw for testing purposes
WANDB_PROJECT="loqa-predictions"
PD_PROMPT_VERSION="zs-v0"
PROMPT_DIR="$REPO_ROOT/Prompts"

# Run
cd "$REPO_ROOT"
echo "Current directory: $(pwd)"

run_prediction() {
    local QG_MODEL_NAME=$1
    local PD_MODEL_NAME=$2
    local PD_MODEL_ACCESS_STRING=$3
    local DATASET_NAME=$4
    local QUESTION_TYPE=$5
    echo "Question Type: $QUESTION_TYPE" "Dataset: $DATASET_NAME" "Split: $SPLIT_NAME" "QG Model: $QG_MODEL_NAME" "PD Model: $PD_MODEL_NAME" "Num Samples: $NUM_SAMPLES"

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
        --hf-token "$HF_TOKEN"
        --wandb-project "$WANDB_PROJECT"
        --wandb-api-key "$WANDB_API_KEY"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --prompt-dir "$PROMPT_DIR"
    )

    [ "$USE_ASYNC" = "1" ] && ARGS+=(--use-async)
    [ "$USE_WANDB" = "1" ] && ARGS+=(--use-wandb)
    [ "$PRINT_PROMPT" = "1" ] && ARGS+=(--print-prompt)

    time python Code/scripts/3_predictions.py "${ARGS[@]}"
    echo "Prediction completed for dataset: $DATASET_NAME, question-type: $QUESTION_TYPE"
}

#Qwen3-4B" "Qwen3-8B" "gpt-4.1-mini" "gpt-oss-120b"

QG_MODEL_NAMES=("gpt-oss-120b")
PD_MODEL_NAMES=("gpt-oss-120b")
PD_MODEL_ACCESS_STRINGS=("openai.gpt-oss-120b")
DATASETS=("DiscourseEE") #"CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT"
QUESTION_TYPES=("optimized_loqa") #"schema" "cot-schema" "loqa" "optimized_loqa"

for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
    for i in "${!PD_MODEL_NAMES[@]}"; do
        PD_MODEL_NAME="${PD_MODEL_NAMES[$i]}"
        PD_MODEL_ACCESS_STRING="${PD_MODEL_ACCESS_STRINGS[$i]}"
        for DATASET in "${DATASETS[@]}"; do
            for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
                echo "QG Model: $QG_MODEL_NAME, PD Model: $PD_MODEL_NAME", "PD Model Access String: $PD_MODEL_ACCESS_STRING"
                echo "Predicting for dataset: $DATASET, Question Type: $QUESTION_TYPE"
                run_prediction "$QG_MODEL_NAME" "$PD_MODEL_NAME" "$PD_MODEL_ACCESS_STRING" "$DATASET" "$QUESTION_TYPE"
            done
        done
    done
done

#for Schema models (QG and PD models are same)
# MODEL_NAMES=("qwen3-4b" "qwen3-8b")
# ACCESS_STRINGS=("Qwen/Qwen3-4B" "Qwen/Qwen3-8B")
# DATASETS=("CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT")
# QUESTION_TYPES=("schema" "cot-schema")

# for i in "${!MODEL_NAMES[@]}"; do
#     QG_MODEL_NAME="${MODEL_NAMES[$i]}"
#     PD_MODEL_NAME="${MODEL_NAMES[$i]}"
#     PD_MODEL_ACCESS_STRING="${ACCESS_STRINGS[$i]}"
#     for DATASET in "${DATASETS[@]}"; do
#         for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
#             echo "QG Model: $QG_MODEL_NAME, PD Model: $PD_MODEL_NAME"
#             echo "Predicting for dataset: $DATASET, Question Type: $QUESTION_TYPE"
#             run_prediction "$QG_MODEL_NAME" "$PD_MODEL_NAME" "$PD_MODEL_ACCESS_STRING" "$DATASET" "$QUESTION_TYPE"
#         done
#     done
# done

##for testing 
# QG_MODEL_NAMES=("gpt-oss-120b")
# PD_MODEL_NAMES=('gpt-oss-120b')
# PD_MODEL_ACCESS_STRINGS=("openai.gpt-oss-120b")
# DATASETS=("DiscourseEE") #"CaseReportBench" "PHEE" "DiscourseEE"
# QUESTION_TYPES=("loqa")

# for QG_MODEL_NAME in "${QG_MODEL_NAMES[@]}"; do
#     for i in "${!PD_MODEL_NAMES[@]}"; do
#         PD_MODEL_NAME="${PD_MODEL_NAMES[$i]}"
#         PD_MODEL_ACCESS_STRING="${PD_MODEL_ACCESS_STRINGS[$i]}"
#         for DATASET in "${DATASETS[@]}"; do
#             for QUESTION_TYPE in "${QUESTION_TYPES[@]}"; do
#                 echo "QG Model: $QG_MODEL_NAME, PD Model: $PD_MODEL_NAME", "PD Model Access String: $PD_MODEL_ACCESS_STRING"
#                 echo "Predicting for dataset: $DATASET, Question Type: $QUESTION_TYPE"
#                 run_prediction "$QG_MODEL_NAME" "$PD_MODEL_NAME" "$PD_MODEL_ACCESS_STRING" "$DATASET" "$QUESTION_TYPE"
#             done
#         done
#     done
# done

#Model names:
# GPT-4.1-mini: openai.gpt-4.1-mini-2025-04-14
# GPT-5-mini: gpt-5-mini-2025-08-07
# GPT-oss-120b: openai.gpt-oss-120b
# Qwen3-32b: qwen.qwen3-vl-32b-instruct-fp8
# Qwen3-8B: Qwen/Qwen3-8B  // vllm server model name
# Qwen3-4B: Qwen/Qwen3-4B