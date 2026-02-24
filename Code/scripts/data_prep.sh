#!/bin/bash
echo "=========================================="
echo "        Data Preparation Script"
echo "=========================================="

# Configuring the path and stuffs
REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
DATASET_NAME="CaseReportBench"  # Options: "CaseReportBench", "DiscourseEE", "PHEE"
DATASET_ROOT="$REPO_ROOT/Dataset"
OUTPUT_ROOT="$REPO_ROOT/Dataset"
PUSH_TO_HUB=true
HUB_REPO_NAME="omar-sharif03/${DATASET_NAME}-processed"
PRIVATE=false

# Build arguments
ARGS=(
    --dataset-root "$DATASET_ROOT"
    --dataset-name "$DATASET_NAME"
    --output-root "$OUTPUT_ROOT"
)

[ "$PUSH_TO_HUB" = true ] && ARGS+=(--push-to-hub --hub-repo-name "$HUB_REPO_NAME")
[ "$PRIVATE" = true ] && ARGS+=(--private)

# Run
cd "$REPO_ROOT"
echo "Dataset: $DATASET_NAME"
echo ""

python Code/scripts/1_data_prep.py "${ARGS[@]}"

echo ""
echo "=========================================="
echo " Data preparation completed!"
echo "=========================================="
