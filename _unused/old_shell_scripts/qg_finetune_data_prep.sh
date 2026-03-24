#!/usr/bin/env bash
set -euo pipefail

# QG finetune data prep wrapper
# Edit values below and run:
#   bash Code/scripts/qg_finetune_data_prep.sh

REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
SCRIPT_PATH="$REPO_ROOT/Code/scripts/7_prepare_qg_finetune_data_from_qo_train.py"
source "$REPO_ROOT/.env"

# Core config
SOURCE_ROOT="$REPO_ROOT/Outputs/qo"
OUTPUT_ROOT="$REPO_ROOT/Outputs/ft_data/qg"
DATASETS=("CaseReportBench" "DiscourseEE" "MACCROBAT" "PHEE")
MODEL_NAME="gpt-oss-120b"
PROMPT_VERSION="zs-v0"
TRAIN_RATIO="0.8"
SEED="7"
SKIP_MISSING_TRAIN_FILES="1"

# Optional prompt override (leave empty to use script default)
PROMPT_TEMPLATE_FILE=""

# Optional per-dataset caps (use -1 for all)
MAX_SAMPLES_PER_DATASET=(
  "CaseReportBench=-1"
  "DiscourseEE=-1"
  "MACCROBAT=-1"
  "PHEE=-1"
)

# Optional merged mixes
# Example format: mixName:DatasetA=1000,DatasetB=1000
MIX_SPECS=(
  "mix_balanced:CaseReportBench=620,PHEE=1860,DiscourseEE=1860,MACCROBAT=1860"
  "mix_high:CaseReportBench=620,PHEE=5000,DiscourseEE=2200,MACCROBAT=5000"
)

ARGS=(
  --source-root "$SOURCE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --datasets "${DATASETS[@]}"
  --model-name "$MODEL_NAME"
  --prompt-version "$PROMPT_VERSION"
  --train-ratio "$TRAIN_RATIO"
  --seed "$SEED"
)

if [[ "$SKIP_MISSING_TRAIN_FILES" == "1" ]]; then
  ARGS+=(--skip-missing-train-files)
else
  ARGS+=(--no-skip-missing-train-files)
fi

if [[ -n "$PROMPT_TEMPLATE_FILE" ]]; then
  ARGS+=(--prompt-template-file "$PROMPT_TEMPLATE_FILE")
fi

if [[ ${#MAX_SAMPLES_PER_DATASET[@]} -gt 0 ]]; then
  ARGS+=(--max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET[@]}")
fi

if [[ ${#MIX_SPECS[@]} -gt 0 ]]; then
  ARGS+=(--mix-specs "${MIX_SPECS[@]}")
fi

cd "$REPO_ROOT"
echo "Running QG finetune data prep..."
echo "python $SCRIPT_PATH ${ARGS[*]}"
python "$SCRIPT_PATH" "${ARGS[@]}"

