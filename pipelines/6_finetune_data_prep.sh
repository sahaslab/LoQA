#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "        6. Fine-tune Data Preparation"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.env"

# ── Configuration ────────────────────────────────────────────
DATASETS=("CaseReportBench" "DiscourseEE" "MACCROBAT" "PHEE")
MODEL_NAME="gpt-oss-120b"
PROMPT_VERSION="zs-v0"
TRAIN_RATIO="0.8"
SEED="7"
SKIP_MISSING_TRAIN_FILES="1"

# Optional merged mixes (format: mixName:DatasetA=N,DatasetB=N)
MIX_SPECS=(
    "mix_balanced:CaseReportBench=620,PHEE=1860,DiscourseEE=1860,MACCROBAT=1860"
)

# Paths
SOURCE_ROOT="$REPO_ROOT/Outputs/qo"
OUTPUT_ROOT="$REPO_ROOT/Outputs/ft_data/qg"
# ─────────────────────────────────────────────────────────────

ARGS=(
    --source-root "$SOURCE_ROOT"
    --output-root "$OUTPUT_ROOT"
    --datasets "${DATASETS[@]}"
    --model-name "$MODEL_NAME"
    --prompt-version "$PROMPT_VERSION"
    --train-ratio "$TRAIN_RATIO"
    --seed "$SEED"
)

[[ "$SKIP_MISSING_TRAIN_FILES" == "1" ]] && ARGS+=(--skip-missing-train-files) || ARGS+=(--no-skip-missing-train-files)
[[ ${#MIX_SPECS[@]} -gt 0 ]] && ARGS+=(--mix-specs "${MIX_SPECS[@]}")

cd "$REPO_ROOT"
echo "Running QG fine-tune data prep..."
python Code/scripts/7_prepare_qg_finetune_data_from_qo_train.py "${ARGS[@]}"
