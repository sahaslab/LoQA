# LoQ

Code and data for **"Improving Information Extraction with Learned Queries"**

Overleaf of the paper: https://www.overleaf.com/read/qfcqjbkbsrtn#0e7d2a

## Project Structure

```
LoQA/
├── Code/
│   ├── scripts/           # Pipeline Python scripts (numbered 1-9)
│   └── src/utils/         # Shared library modules
├── pipelines/             # Shell scripts to run each pipeline stage
├── Dataset/               # JSON data splits and schemas per dataset
├── Prompts/               # Prompt templates (qg/, pd/, qo/, lc/, dq/)
├── xDev/                  # Notebooks and analysis
└── Outputs/               # Generated artifacts (gitignored)
```

## Setup

```bash
# 1. Clone the repository
git clone <repo-url> && cd LoQA

# 2. Create and activate a conda environment (requires GPU for vLLM)
conda create -n loqa python=3.11 -y
conda activate loqa

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (used in evaluation)
python -m spacy download en_core_web_sm

# 5. Configure API keys
cp .env.example .env
# Edit .env with your credentials (see .env.example for required keys)
```

> **Note:** This project requires a GPU environment for vLLM-based inference. Make sure CUDA is available in your conda environment.

## Pipeline

The pipeline has two tracks: **core extraction** and **question optimization / fine-tuning**.

### Core Pipeline (QG → Prediction → Evaluation)

```bash
# Run all three stages end-to-end:
bash pipelines/run_core_pipeline.sh

# Or run stages individually:
bash pipelines/1_data_prep.sh              # Prepare dataset splits
bash pipelines/2_question_generation.sh    # Generate questions (schema/loqa/dynamicQ)
bash pipelines/3_prediction.sh             # Extract arguments using questions
bash pipelines/4_evaluation.sh             # Evaluate + score (EM/RM/CM)
```

### Question Optimization

```bash
bash pipelines/5_question_optimization.sh  # Iterative question refinement loop
```

### Fine-tuning (QG model via Tinker)

```bash
bash pipelines/6_finetune_data_prep.sh     # Prepare SFT data from QO outputs
bash pipelines/7_finetune_train.sh         # LoRA fine-tune with Tinker
bash pipelines/8_finetune_generate.sh      # Generate with fine-tuned QG model
bash pipelines/9_finetune_full_pipeline.sh # Full FT-QG → PD → Eval pipeline
```

Edit the configuration variables at the top of each shell script to customize datasets, models, and parameters.

## Datasets

| Dataset | Domain |
|---------|--------|
| DiscourseEE | Discourse-level event extraction |
| PHEE | Pharmacovigilance event extraction |
| CaseReportBench | Clinical case report extraction |
| MACCROBAT | Medical argument extraction |

## Question Types

| Type | Description |
|------|-------------|
| `schema` | Knowledge questions derived from role schema |
| `cot-schema` | Chain-of-thought schema questions |
| `loqa` | LLM-generated questions (GT visible during generation) |
| `dynamicQ` | Zero-shot LLM-generated questions (no GT) |
| `optimized_loqa` | Questions refined via the QO loop |
| `ft` | Questions from a fine-tuned QG model |

## Evaluation Metrics

- **EM** — Exact match
- **RM** — Relaxed match (semantic similarity ≥ 0.85)
- **CM** — Complex match (LLM-as-judge)
