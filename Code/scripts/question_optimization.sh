#!/bin/bash

echo "=========================================="
echo "      Question Optimization Script"
echo "=========================================="

# Configuring the path and environment
REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
source "$REPO_ROOT/.env"

# Common Variables - Modify these as needed
DATASET_NAME="CaseReportBench"  # "DiscourseEE" "PHEE" "CaseReportBench"
SPLIT_NAME="train"               # "train" "test" "dev"
NUM_SAMPLES=10000              # -1 = all valid samples

# Question optimization model
QO_MODEL_NAME="gpt-oss-120b"
QO_MODEL_ORIGIN="dartmouth"
QO_MODEL_ACCESS_STRING="openai.gpt-oss-120b"
QO_TEMPERATURE=0.0
QO_GPU_UTIL=0.9
QO_REASONING_EFFORT="none"
QO_PROMPT_VERSION="zs-v0"
INITIAL_QO_PROMPT_VERSION="zs-v0"

# Prediction model (for argument extraction)
PD_MODEL_NAME="gpt-oss-120b"
PD_MODEL_ORIGIN="dartmouth"
PD_MODEL_ACCESS_STRING="openai.gpt-oss-120b"
PD_TEMPERATURE=0.0
PD_GPU_UTIL=0.9
PD_REASONING_EFFORT="none"
PD_PROMPT_VERSION="zs-v0"
VERBOSE="1" # set to "1" to print progress and results to stdout
PRINT_PROMPT="" # set to "1" to print formatted prompt before each get_response (initial QG, arg extraction, refinement)

# Refinement loop
NUM_ITERATIONS=5
TARGET_SCORE=1.0
MAXIMUM_PATIENCE=3

#parallel execution
PARALLEL="" # set to "1" to run refinement loop in parallel
MAX_WORKERS=4 # maximum number of workers for parallel execution

# Paths (usually do not change)
DATASET_ROOT="$REPO_ROOT/Dataset"
OUTPUT_PATH="$REPO_ROOT/Outputs/qo/"
PROMPT_DIR="$REPO_ROOT/Prompts"
CACHE_DIR="/dartfs-hpc/rc/home/j/f006f3j/lab/shared"

# Log directory
LOG_DIR="$REPO_ROOT/Outputs/logs"
mkdir -p "$LOG_DIR"

# Add date and time to log file names for uniqueness
CURRENT_DATETIME=$(date +"%Y-%m-%d")
LOG_DIR_WITH_DATE="$LOG_DIR/$CURRENT_DATETIME"
mkdir -p "$LOG_DIR_WITH_DATE"
LOG_DIR="$LOG_DIR_WITH_DATE"

# Build and run
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
    [ "$VERBOSE" = "1" ] && ARGS+=(--verbose)
    [ "$PRINT_PROMPT" = "1" ] && ARGS+=(--print-prompt)
    if [ "$PARALLEL" = "1" ]; then
        ARGS+=(--parallel --max-workers "$MAX_WORKERS")
    fi

    cd "$REPO_ROOT"
    echo "Current directory: $(pwd)"
    echo "Dataset: $DATASET_NAME, Split: $SPLIT_NAME, Number of samples: $NUM_SAMPLES"
    echo ""
    time python Code/scripts/6_q_optimization.py "${ARGS[@]}"
}

# Run for configured datasets
DATASETS=("PHEE")  # "CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT"

for DATASET in "${DATASETS[@]}"; do    
    LOG_FILE="$LOG_DIR/qo-${QO_MODEL_NAME}-${DATASET}-${SPLIT_NAME}-${PD_MODEL_NAME}.out"
    echo "Logging to: $LOG_FILE"
    echo "QO Model: $QO_MODEL_NAME, PD Model: $PD_MODEL_NAME"
    echo "Running question optimization for dataset: $DATASET, Number of samples: $NUM_SAMPLES"
    run_question_optimization "$DATASET" 2>&1 | tee "$LOG_FILE"
done
