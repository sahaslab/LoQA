#!/bin/bash
echo "=========================================="
echo "        Test Evaluation Script"
echo "=========================================="

REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
source "$REPO_ROOT/.env"

# Change to repo root to ensure relative paths work
cd "$REPO_ROOT"

# Parameters (match test_evaluation.py defaults)
DATASET_NAME="MACCROBAT"
RM_THRESHOLD=0.85
VERBOSE=true
NUM_SAMPLES=-1

# Log directory
LOG_DIR="$REPO_ROOT/Outputs/logs"
CURRENT_DATETIME=$(date +"%Y-%m-%d")
LOG_DIR="$LOG_DIR/$CURRENT_DATETIME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_evaluation-${DATASET_NAME}.out"

echo "Dataset: $DATASET_NAME"
echo "RM threshold: $RM_THRESHOLD"
echo "Logging to: $LOG_FILE"
echo ""

ARGS=(
    --dataset-name "$DATASET_NAME"
    --rm-threshold "$RM_THRESHOLD"
    --num-samples "$NUM_SAMPLES"
)
[[ "$VERBOSE" == true ]] && ARGS+=(--verbose)

time python Code/scripts/test_evaluation.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
