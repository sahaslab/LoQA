#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        1. Data Preparation"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Configuration ────────────────────────────────────────────
DATASET_NAME="CaseReportBench"          # CaseReportBench | DiscourseEE | PHEE | MACCROBAT
DATASET_ROOT="$REPO_ROOT/Dataset"
OUTPUT_ROOT="$REPO_ROOT/Dataset"
PUSH_TO_HUB=false
HUB_REPO_NAME="omar-sharif03/${DATASET_NAME}-processed"
PRIVATE=false
# ─────────────────────────────────────────────────────────────

ARGS=(
    --dataset-root "$DATASET_ROOT"
    --dataset-name "$DATASET_NAME"
    --output-root  "$OUTPUT_ROOT"
)

[[ "$PUSH_TO_HUB" == true ]] && ARGS+=(--push-to-hub --hub-repo-name "$HUB_REPO_NAME")
[[ "$PRIVATE"     == true ]] && ARGS+=(--private)

cd "$REPO_ROOT"
echo "Dataset: $DATASET_NAME"
python Code/scripts/1_data_prep.py "${ARGS[@]}"
echo "Data preparation completed."
