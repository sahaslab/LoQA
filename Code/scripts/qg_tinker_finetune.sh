#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
SCRIPT_PATH="$REPO_ROOT/Code/scripts/8_tinker_qg_finetune.py"

# Load .env if present (TINKER_API_KEY, WANDB_API_KEY)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is not set."
  exit 1
fi

# TRAIN_FILE="$REPO_ROOT/Outputs/ft_data/qg/mixes/mix_balanced/train.sft.jsonl"
# VAL_FILE="$REPO_ROOT/Outputs/ft_data/qg/mixes/mix_balanced/val.sft.jsonl"
TRAIN_FILE="$REPO_ROOT/Outputs/ft_data/qg/MACCROBAT/train.sft.jsonl"
VAL_FILE="$REPO_ROOT/Outputs/ft_data/qg/MACCROBAT/val.sft.jsonl"
OUTPUT_DIR="$REPO_ROOT/Outputs/ft_runs/"
RUN_NAME="qwen3-4b-MACCROBAT-v1"

BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
# BASE_MODEL="Qwen/Qwen3-8B"
RANK="32"
SEED="7"
BATCH_SIZE="32" # for higher dataset size using batch size 32. using 16 for smaller dataset.
VAL_BATCH_SIZE="64" # increase for faster evaluation should not be an issue
NUM_EPOCHS="2"
MAX_SEQ_LEN="4096"
LEARNING_RATE="3.7e-4" # qwen sugested it
EVAL_EVERY_STEPS="40"
WANDB_PROJECT="loqa-qg-finetune"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[INFO] Run name      : $RUN_NAME"
echo "[INFO] Base model    : $BASE_MODEL"
echo "[INFO] Epochs        : $NUM_EPOCHS"
echo "[INFO] Learning rate : $LEARNING_RATE"
echo "[INFO] Batch size    : $BATCH_SIZE"

cd "$REPO_ROOT"
python "$SCRIPT_PATH" \
  --train-file        "$TRAIN_FILE" \
  --val-file          "$VAL_FILE" \
  --output-dir        "$OUTPUT_DIR" \
  --run-name          "$RUN_NAME" \
  --base-model        "$BASE_MODEL" \
  --rank              "$RANK" \
  --seed              "$SEED" \
  --batch-size        "$BATCH_SIZE" \
  --val-batch-size    "$VAL_BATCH_SIZE" \
  --num-epochs        "$NUM_EPOCHS" \
  --max-seq-len       "$MAX_SEQ_LEN" \
  --learning-rate     "$LEARNING_RATE" \
  --eval-every-steps  "$EVAL_EVERY_STEPS" \
  --wandb-project     "$WANDB_PROJECT"
