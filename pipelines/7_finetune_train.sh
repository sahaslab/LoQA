#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        7. Fine-tune QG Model (Tinker)"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "ERROR: TINKER_API_KEY not set."; exit 1; }

# ── Configuration ────────────────────────────────────────────
TRAIN_FILE="$REPO_ROOT/Outputs/ft_data/qg/mixes/mix_balanced/train.sft.jsonl"
VAL_FILE="$REPO_ROOT/Outputs/ft_data/qg/mixes/mix_balanced/val.sft.jsonl"
OUTPUT_DIR="$REPO_ROOT/Outputs/ft_runs"
RUN_NAME="qwen3-8b-mix-balanced-v1"

BASE_MODEL="Qwen/Qwen3-8B"
RANK="32"
SEED="7"
BATCH_SIZE="32"
VAL_BATCH_SIZE="64"
NUM_EPOCHS="2"
MAX_SEQ_LEN="4096"
LEARNING_RATE="3.7e-4"
EVAL_EVERY_STEPS="40"
WANDB_PROJECT="loqa-qg-finetune"
# ─────────────────────────────────────────────────────────────

echo "[INFO] Run: $RUN_NAME | Base: $BASE_MODEL | Epochs: $NUM_EPOCHS | LR: $LEARNING_RATE"

cd "$REPO_ROOT"
python Code/scripts/8_tinker_qg_finetune.py \
    --train-file       "$TRAIN_FILE" \
    --val-file         "$VAL_FILE" \
    --output-dir       "$OUTPUT_DIR" \
    --run-name         "$RUN_NAME" \
    --base-model       "$BASE_MODEL" \
    --rank             "$RANK" \
    --seed             "$SEED" \
    --batch-size       "$BATCH_SIZE" \
    --val-batch-size   "$VAL_BATCH_SIZE" \
    --num-epochs       "$NUM_EPOCHS" \
    --max-seq-len      "$MAX_SEQ_LEN" \
    --learning-rate    "$LEARNING_RATE" \
    --eval-every-steps "$EVAL_EVERY_STEPS" \
    --wandb-project    "$WANDB_PROJECT"
