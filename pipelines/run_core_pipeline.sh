#!/usr/bin/env bash
set -euo pipefail

#
# Master script: runs the core LoQA pipeline end-to-end.
#   Question Generation → Prediction → Evaluation & Scoring
#
# Usage:
#   bash pipelines/run_core_pipeline.sh
#
# Customize each step by editing the individual pipeline scripts,
# or run them independently:
#   bash pipelines/2_question_generation.sh
#   bash pipelines/3_prediction.sh
#   bash pipelines/4_evaluation.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  LoQA Core Pipeline"
echo "  QG → Prediction → Evaluation & Scoring"
echo "============================================================"
echo ""

echo "──── Step 1/3: Question Generation ────"
bash "$SCRIPT_DIR/2_question_generation.sh"
echo ""

echo "──── Step 2/3: Prediction ────"
bash "$SCRIPT_DIR/3_prediction.sh"
echo ""

echo "──── Step 3/3: Evaluation & Scoring ────"
bash "$SCRIPT_DIR/4_evaluation.sh"
echo ""

echo "============================================================"
echo "  Pipeline complete."
echo "============================================================"
