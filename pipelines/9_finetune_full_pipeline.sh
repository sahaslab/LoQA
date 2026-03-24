#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " 9. Full Fine-tune Pipeline"
echo "    FT-QG → Prediction → Eval → Scoring"
echo "=========================================="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "ERROR: TINKER_API_KEY not set."; exit 1; }

# ── Configuration ────────────────────────────────────────────
DATASET_ROOT="$REPO_ROOT/Dataset"
SPLITS=("test")
QUESTION_TYPE="ft"
QG_PROMPT_VERSION="zs-v0"
PD_PROMPT_VERSION="zs-v0"

# Fine-tuned QG models (auto-generated from base × dataset)
QG_BASE_MODELS=("qwen3-8b")
QG_DATASETS=("CaseReportBench" "PHEE" "DiscourseEE" "MACCROBAT")
QG_VERSION="v1"

DATASETS=()
QG_MODEL_NAMES=()
QG_RUN_DIRS=()
for base in "${QG_BASE_MODELS[@]}"; do
    for ds in "${QG_DATASETS[@]}"; do
        DATASETS+=("$ds")
        QG_MODEL_NAMES+=("${base}-${ds}-${QG_VERSION}")
        QG_RUN_DIRS+=("$REPO_ROOT/Outputs/ft_runs/${base}-${ds}-${QG_VERSION}")
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
QG_N_SAMPLES="-1"
QG_MAX_CONCURRENT="8"

# PD model (index-aligned arrays)
PD_MODEL_NAMES=("gpt-oss-120b")
PD_MODEL_ORIGINS=("dartmouth")
PD_MODEL_ACCESS_STRINGS=("openai.gpt-oss-120b")
PD_REASONING_EFFORTS=("none")
PD_TEMPERATURE="0.0"
PD_GPU_UTIL="0.9"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache}"
PRED_NUM_SAMPLES="-1"
USE_ASYNC="1"
BATCH_SIZE="100"

# Evaluation params
RM_THRESHOLD="0.85"
DO_COMPLEX_MATCH="1"
VERBOSE_EVAL="1"
EVAL_NUM_SAMPLES="-1"

# Scoring params
OVERALL_SCORES="1"
SAVE_SCORES="1"
VERBOSE_SCORE="1"

# Logging
LOG_DIR="$REPO_ROOT/Outputs/logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_DIR"
# ─────────────────────────────────────────────────────────────

mkdir -p "$QG_OUTPUT_PATH" "$PD_OUTPUT_PATH" "$EVALUATION_PATH" "$SCORES_PATH"

run_qg_generation() {
    local dataset=$1 split=$2 qg_model=$3 run_dir=$4
    echo "[QG] dataset=$dataset split=$split model=$qg_model"
    local args=(
        --dataset-root "$DATASET_ROOT" --dataset-name "$dataset"
        --split-name "$split" --question-type "$QUESTION_TYPE"
        --qg-model-name "$qg_model" --qg-prompt-version "$QG_PROMPT_VERSION"
        --run-dir "$run_dir" --output-path "$QG_OUTPUT_PATH"
        --max-new-tokens "$QG_MAX_NEW_TOKENS" --temperature "$QG_TEMPERATURE"
        --top-p "$QG_TOP_P" --max-concurrent "$QG_MAX_CONCURRENT"
    )
    [[ "$QG_N_SAMPLES" != "-1" ]] && args+=(--n-samples "$QG_N_SAMPLES")
    time python "$REPO_ROOT/Code/scripts/9_tinker_qg_generate_async.py" "${args[@]}" \
        2>&1 | tee "$LOG_DIR/qg-${qg_model}-${dataset}-${split}.out"
}

run_prediction() {
    local dataset=$1 split=$2 qg_model=$3 pd_model=$4
    local pd_origin=$5 pd_access=$6 reasoning=$7
    echo "[PD] dataset=$dataset qg=$qg_model pd=$pd_model"
    local args=(
        --dataset-name "$dataset" --split-name "$split"
        --question-type "$QUESTION_TYPE"
        --qg-model-name "$qg_model" --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-model-origin "$pd_origin" --pd-model-name "$pd_model"
        --pd-model-access-string "$pd_access"
        --pd-temperature "$PD_TEMPERATURE" --pd-gpu-util "$PD_GPU_UTIL"
        --cache-dir "$CACHE_DIR" --reasoning-effort "$reasoning"
        --num-samples "$PRED_NUM_SAMPLES"
        --qg-output-path "$QG_OUTPUT_PATH" --pd-output-path "$PD_OUTPUT_PATH"
        --batch-size "$BATCH_SIZE" --hf-token "${HF_TOKEN:-}"
        --pd-prompt-version "$PD_PROMPT_VERSION" --prompt-dir "$REPO_ROOT/Prompts"
    )
    [[ "$USE_ASYNC" == "1" ]] && args+=(--use-async)
    time python "$REPO_ROOT/Code/scripts/3_predictions.py" "${args[@]}" \
        2>&1 | tee "$LOG_DIR/pd-${qg_model}-${pd_model}-${dataset}-${split}.out"
}

run_evaluation() {
    local dataset=$1 split=$2 qg_model=$3 pd_model=$4
    echo "[EV] dataset=$dataset qg=$qg_model pd=$pd_model"
    local args=(
        --dataset-name "$dataset" --split-name "$split"
        --question-type "$QUESTION_TYPE"
        --qg-model-name "$qg_model" --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-path "$PD_OUTPUT_PATH" --pd-model-name "$pd_model"
        --pd-prompt-version "$PD_PROMPT_VERSION"
        --rm-threshold "$RM_THRESHOLD" --evaluation-path "$EVALUATION_PATH"
        --num-samples "$EVAL_NUM_SAMPLES"
    )
    [[ "$DO_COMPLEX_MATCH" == "1" ]] && args+=(--do-complex-match)
    [[ "$VERBOSE_EVAL"     == "1" ]] && args+=(--verbose)
    time python "$REPO_ROOT/Code/scripts/4_evaluation.py" "${args[@]}" \
        2>&1 | tee "$LOG_DIR/ev-${qg_model}-${pd_model}-${dataset}-${split}.out"
}

run_scoring() {
    local dataset=$1 split=$2 qg_model=$3 pd_model=$4
    echo "[SC] dataset=$dataset qg=$qg_model pd=$pd_model"
    local args=(
        --dataset-name "$dataset" --split-name "$split"
        --pred-key "$QUESTION_TYPE"
        --qg-model-name "$qg_model" --qg-prompt-version "$QG_PROMPT_VERSION"
        --pd-model-name "$pd_model" --pd-prompt-version "$PD_PROMPT_VERSION"
        --rm-threshold "$RM_THRESHOLD"
        --evaluation-path "$EVALUATION_PATH" --scores-path "$SCORES_PATH"
    )
    [[ "$DO_COMPLEX_MATCH" == "1" ]] && args+=(--do-complex-match)
    [[ "$OVERALL_SCORES"   == "1" ]] && args+=(--overall-scores)
    [[ "$SAVE_SCORES"      == "1" ]] && args+=(--save-scores)
    [[ "$VERBOSE_SCORE"    == "1" ]] && args+=(--verbose)
    time python "$REPO_ROOT/Code/scripts/5_scoring.py" "${args[@]}" \
        2>&1 | tee "$LOG_DIR/sc-${qg_model}-${pd_model}-${dataset}-${split}.out"
}

# ── Pipeline ─────────────────────────────────────────────────
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Start: $(date)"

cd "$REPO_ROOT"
for split in "${SPLITS[@]}"; do
    for qg_idx in "${!QG_MODEL_NAMES[@]}"; do
        qg_model="${QG_MODEL_NAMES[$qg_idx]}"
        dataset="${DATASETS[$qg_idx]}"
        run_dir="${QG_RUN_DIRS[$qg_idx]}"

        run_qg_generation "$dataset" "$split" "$qg_model" "$run_dir"

        for pd_idx in "${!PD_MODEL_NAMES[@]}"; do
            run_prediction "$dataset" "$split" "$qg_model" \
                "${PD_MODEL_NAMES[$pd_idx]}" "${PD_MODEL_ORIGINS[$pd_idx]}" \
                "${PD_MODEL_ACCESS_STRINGS[$pd_idx]}" "${PD_REASONING_EFFORTS[$pd_idx]}"
            run_evaluation "$dataset" "$split" "$qg_model" "${PD_MODEL_NAMES[$pd_idx]}"
            run_scoring    "$dataset" "$split" "$qg_model" "${PD_MODEL_NAMES[$pd_idx]}"
        done
    done
done

echo "[INFO] End: $(date)"
echo "[DONE] Full fine-tune pipeline finished."
