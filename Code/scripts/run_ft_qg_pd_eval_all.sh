#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " FT-QG -> Prediction -> Evaluation Runner "
echo "=========================================="

REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
cd "$REPO_ROOT"

# Load environment variables (TINKER_API_KEY, HF_TOKEN, WANDB_API_KEY, etc.)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

# Dataset / splits
DATASET_ROOT="$REPO_ROOT/Dataset"
SPLITS=("test")

# Shared question type for generated questions and downstream stages.
# Keep "ft" unless your files are intentionally keyed differently.
QUESTION_TYPE="ft"
QG_PROMPT_VERSION="zs-v0"
PD_PROMPT_VERSION="zs-v0"

# Fine-tuned QG models — auto-generated from base models × datasets
QG_BASE_MODELS=("qwen3-8b" "qwen3-4b") #"qwen3-4b"
QG_DATASETS=("MACCROBAT" "DiscourseEE" "CaseReportBench" "PHEE") #"DiscourseEE" "CaseReportBench" "PHEE")
QG_VERSION="v1"

DATASETS=()
QG_MODEL_NAMES=()
QG_RUN_DIRS=()
for base in "${QG_BASE_MODELS[@]}"; do
  for ds in "${QG_DATASETS[@]}"; do
    qg_name="${base}-${ds}-${QG_VERSION}"
    DATASETS+=("$ds")
    QG_MODEL_NAMES+=("$qg_name")
    QG_RUN_DIRS+=("$REPO_ROOT/Outputs/ft_runs/$qg_name")
  done
done

# Output locations
QG_OUTPUT_PATH="$REPO_ROOT/Outputs/ft_outputs/qg"
PD_OUTPUT_PATH="$REPO_ROOT/Outputs/ft_outputs/pd"
EVALUATION_PATH="$REPO_ROOT/Outputs/ft_outputs/ev"
SCORES_PATH="$REPO_ROOT/Outputs/ft_outputs/sc"

# QG generation params
QG_MAX_NEW_TOKENS="512"
QG_TEMPERATURE="0.7"
QG_TOP_P="0.8"
QG_N_SAMPLES="-1"      # -1 means all rows
QG_MAX_CONCURRENT="8"

# Prediction model configs (index-aligned arrays)
#   model name (for filenames)   |  origin       |  access string              |  reasoning effort
# PD_MODEL_NAMES=(    'gpt-oss-120b'      "qwen3-8b"          "qwen3-4b"          "gpt-5-mini-medium"           "gemini-3.1-pro-high"       )
# PD_MODEL_ORIGINS=(        "dartmouth"        "vllm-local"        "vllm-local"        "openai"                   "google"                 )
# PD_MODEL_ACCESS_STRINGS=( "openai.gpt-oss-120b"    "Qwen/Qwen3-8B"    "Qwen/Qwen3-4B"    "gpt-5-mini-2025-08-07"       "gemini-3.1-pro-preview"    )
# PD_REASONING_EFFORTS=(   "none"            "medium"            "medium"            "medium"                      "high"                      )

PD_MODEL_NAMES=( "gpt-oss-120b"       )
PD_MODEL_ORIGINS=(   "dartmouth"   )
PD_MODEL_ACCESS_STRINGS=(  "openai.gpt-oss-120b"    )
PD_REASONING_EFFORTS=(  "none"  )

# Prediction params
PD_TEMPERATURE="0.0"
PD_GPU_UTIL="0.9"
CACHE_DIR="/dartfs-hpc/rc/home/j/f006f3j/lab/shared"
PRED_NUM_SAMPLES="-1"  # -1 means all rows
USE_ASYNC="1"          # "1" to enable async in prediction
BATCH_SIZE="100"
PRINT_PROMPT=""        # set "1" if you want prompt printout
USE_WANDB_PRED=""      # set "1" to log predictions to wandb
WANDB_PRED_PROJECT="loqa-predictions"

# Evaluation params
RM_THRESHOLD="0.85"
DO_COMPLEX_MATCH="1"   # "1" to enable LLM-as-judge stage in evaluation
VERBOSE_EVAL="1"       # "1" for verbose evaluation output
EVAL_NUM_SAMPLES="-1"

# Scoring params
OVERALL_SCORES="1"
ROLE_WISE_SCORES=""    # set "1" for role-wise scores
SAVE_SCORES="1"
USE_WANDB_SCORE=""     # set "1" to log scores to wandb
WANDB_SCORE_PROJECT="loqa-scores"
VERBOSE_SCORE="1"

# Logging
LOG_ROOT="$REPO_ROOT/Outputs/logs"
RUN_STAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
LOG_DIR="$LOG_ROOT/$RUN_STAMP"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ "${#QG_MODEL_NAMES[@]}" -ne "${#QG_RUN_DIRS[@]}" ]]; then
  echo "ERROR: QG_MODEL_NAMES and QG_RUN_DIRS must have the same length."
  exit 1
fi

if [[ "${#PD_MODEL_NAMES[@]}" -ne "${#PD_MODEL_ORIGINS[@]}" ]] || \
   [[ "${#PD_MODEL_NAMES[@]}" -ne "${#PD_MODEL_ACCESS_STRINGS[@]}" ]] || \
   [[ "${#PD_MODEL_NAMES[@]}" -ne "${#PD_REASONING_EFFORTS[@]}" ]]; then
  echo "ERROR: PD model arrays (NAMES, ORIGINS, ACCESS_STRINGS, REASONING_EFFORTS) must be index-aligned and same length."
  exit 1
fi

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is not set. Required for fine-tuned QG generation."
  exit 1
fi

mkdir -p "$QG_OUTPUT_PATH" "$PD_OUTPUT_PATH" "$EVALUATION_PATH" "$SCORES_PATH"

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

run_qg_generation() {
  local dataset_name="$1"
  local split_name="$2"
  local qg_model_name="$3"
  local run_dir="$4"

  local log_file="$LOG_DIR/qg-${qg_model_name}-${dataset_name}-${split_name}.out"
  echo "[QG] dataset=$dataset_name split=$split_name qg_model=$qg_model_name"
  echo "[QG] log: $log_file"

  local args=(
    --dataset-root      "$DATASET_ROOT"
    --dataset-name      "$dataset_name"
    --split-name        "$split_name"
    --question-type     "$QUESTION_TYPE"
    --qg-model-name     "$qg_model_name"
    --qg-prompt-version "$QG_PROMPT_VERSION"
    --run-dir           "$run_dir"
    --output-path       "$QG_OUTPUT_PATH"
    --max-new-tokens    "$QG_MAX_NEW_TOKENS"
    --temperature       "$QG_TEMPERATURE"
    --top-p             "$QG_TOP_P"
    --max-concurrent    "$QG_MAX_CONCURRENT"
  )

  if [[ "$QG_N_SAMPLES" != "-1" ]]; then
    args+=(--n-samples "$QG_N_SAMPLES")
  fi

  time python "$REPO_ROOT/Code/scripts/9_tinker_qg_generate_async.py" "${args[@]}" 2>&1 | tee "$log_file"
}

run_prediction() {
  local dataset_name="$1"
  local split_name="$2"
  local qg_model_name="$3"
  local pd_model_name="$4"
  local pd_model_origin="$5"
  local pd_model_access_string="$6"
  local reasoning_effort="$7"

  local log_file="$LOG_DIR/pd-${qg_model_name}-${pd_model_name}-${dataset_name}-${split_name}.out"
  echo "[PD] dataset=$dataset_name split=$split_name qg_model=$qg_model_name pd_model=$pd_model_name reasoning=$reasoning_effort"
  echo "[PD] log: $log_file"

  local args=(
    --dataset-name            "$dataset_name"
    --split-name              "$split_name"
    --question-type           "$QUESTION_TYPE"
    --qg-model-name           "$qg_model_name"
    --qg-prompt-version       "$QG_PROMPT_VERSION"
    --pd-model-origin         "$pd_model_origin"
    --pd-model-name           "$pd_model_name"
    --pd-model-access-string  "$pd_model_access_string"
    --pd-temperature          "$PD_TEMPERATURE"
    --pd-gpu-util             "$PD_GPU_UTIL"
    --cache-dir               "$CACHE_DIR"
    --reasoning-effort        "$reasoning_effort"
    --num-samples             "$PRED_NUM_SAMPLES"
    --qg-output-path          "$QG_OUTPUT_PATH"
    --pd-output-path          "$PD_OUTPUT_PATH"
    --batch-size              "$BATCH_SIZE"
    --hf-token                "${HF_TOKEN:-}"
    --wandb-project           "$WANDB_PRED_PROJECT"
    --wandb-api-key           "${WANDB_API_KEY:-}"
    --pd-prompt-version       "$PD_PROMPT_VERSION"
    --prompt-dir              "$REPO_ROOT/Prompts"
  )

  [[ "$USE_ASYNC" == "1" ]] && args+=(--use-async)
  [[ "$USE_WANDB_PRED" == "1" ]] && args+=(--use-wandb)
  [[ "$PRINT_PROMPT" == "1" ]] && args+=(--print-prompt)

  time python "$REPO_ROOT/Code/scripts/3_predictions.py" "${args[@]}" 2>&1 | tee "$log_file"
}

run_evaluation() {
  local dataset_name="$1"
  local split_name="$2"
  local qg_model_name="$3"
  local pd_model_name="$4"

  local log_file="$LOG_DIR/ev-${qg_model_name}-${pd_model_name}-${dataset_name}-${split_name}.out"
  echo "[EV] dataset=$dataset_name split=$split_name qg_model=$qg_model_name pd_model=$pd_model_name"
  echo "[EV] log: $log_file"

  local args=(
    --dataset-name       "$dataset_name"
    --split-name         "$split_name"
    --question-type      "$QUESTION_TYPE"
    --qg-model-name      "$qg_model_name"
    --qg-prompt-version  "$QG_PROMPT_VERSION"
    --pd-path            "$PD_OUTPUT_PATH"
    --pd-model-name      "$pd_model_name"
    --pd-prompt-version  "$PD_PROMPT_VERSION"
    --rm-threshold       "$RM_THRESHOLD"
    --evaluation-path    "$EVALUATION_PATH"
    --num-samples        "$EVAL_NUM_SAMPLES"
  )

  [[ "$DO_COMPLEX_MATCH" == "1" ]] && args+=(--do-complex-match)
  [[ "$VERBOSE_EVAL" == "1" ]] && args+=(--verbose)

  time python "$REPO_ROOT/Code/scripts/4_evaluation.py" "${args[@]}" 2>&1 | tee "$log_file"
}

run_scoring() {
  local dataset_name="$1"
  local split_name="$2"
  local qg_model_name="$3"
  local pd_model_name="$4"

  local log_file="$LOG_DIR/sc-${qg_model_name}-${pd_model_name}-${dataset_name}-${split_name}.out"
  echo "[SC] dataset=$dataset_name split=$split_name qg_model=$qg_model_name pd_model=$pd_model_name"
  echo "[SC] log: $log_file"

  local args=(
    --dataset-name       "$dataset_name"
    --split-name         "$split_name"
    --pred-key           "$QUESTION_TYPE"
    --qg-model-name      "$qg_model_name"
    --qg-prompt-version  "$QG_PROMPT_VERSION"
    --pd-model-name      "$pd_model_name"
    --pd-prompt-version  "$PD_PROMPT_VERSION"
    --rm-threshold       "$RM_THRESHOLD"
    --wandb-project      "$WANDB_SCORE_PROJECT"
    --evaluation-path    "$EVALUATION_PATH"
    --scores-path        "$SCORES_PATH"
  )

  [[ "$DO_COMPLEX_MATCH" == "1" ]] && args+=(--do-complex-match)
  [[ "$OVERALL_SCORES" == "1" ]] && args+=(--overall-scores)
  [[ "$ROLE_WISE_SCORES" == "1" ]] && args+=(--role-wise-scores)
  [[ "$SAVE_SCORES" == "1" ]] && args+=(--save-scores)
  [[ "$USE_WANDB_SCORE" == "1" ]] && args+=(--use-wandb --wandb-api-key "${WANDB_API_KEY:-}")
  [[ "$VERBOSE_SCORE" == "1" ]] && args+=(--verbose)

  time python "$REPO_ROOT/Code/scripts/5_scoring.py" "${args[@]}" 2>&1 | tee "$log_file"
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Start: $(date)"

for split in "${SPLITS[@]}"; do
  for qg_idx in "${!QG_MODEL_NAMES[@]}"; do
    qg_model_name="${QG_MODEL_NAMES[$qg_idx]}"
    qg_run_dir="${QG_RUN_DIRS[$qg_idx]}"
    dataset_name="${DATASETS[$qg_idx]}"
    # Run QG once per (dataset, split, qg_model)
    run_qg_generation "$dataset_name" "$split" "$qg_model_name" "$qg_run_dir"

    # Then run prediction + evaluation + scoring for all PD models
    for pd_idx in "${!PD_MODEL_NAMES[@]}"; do
        pd_model_name="${PD_MODEL_NAMES[$pd_idx]}"
        pd_model_origin="${PD_MODEL_ORIGINS[$pd_idx]}"
        pd_model_access_string="${PD_MODEL_ACCESS_STRINGS[$pd_idx]}"
        reasoning_effort="${PD_REASONING_EFFORTS[$pd_idx]}"

        run_prediction "$dataset_name" "$split" "$qg_model_name" "$pd_model_name" "$pd_model_origin" "$pd_model_access_string" "$reasoning_effort"
        run_evaluation "$dataset_name" "$split" "$qg_model_name" "$pd_model_name"
        run_scoring "$dataset_name" "$split" "$qg_model_name" "$pd_model_name"
    done
  done
done

echo "[INFO] End: $(date)"
echo "[DONE] Full pipeline finished."
