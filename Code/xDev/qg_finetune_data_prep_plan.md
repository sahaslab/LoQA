# QG Fine-Tuning Data Prep (Paused Notes)

## Current Understanding

Goal:
- Build fine-tuning data for a question-generator model.
- Use only `train` split files from `Outputs/qo` as source.
- Do **not** modify source `dev`/`test` files.
- Create new fine-tuning splits from source train data: `train/val`.
- Support:
  - per-dataset datasets (separate outputs),
  - merged dataset mixes,
  - custom sample count per dataset for each mix.

## Source Check Done

Detected `Outputs/qo` dataset folders:
- `CaseReportBench`
- `DiscourseEE`
- `MACCROBAT`
- `PHEE`

Train files currently available:
- `CaseReportBench`: yes
- `DiscourseEE`: yes
- `PHEE`: yes
- `MACCROBAT`: yes

## Planned Script (Not Implemented Yet)

Proposed new script:
- `Code/scripts/7_prepare_qg_finetune_data_from_qo_train.py`

Behavior:
1. Read only files matching:
   - `Outputs/qo/<DATASET>/optimized_loqa-<DATASET>-train-<MODEL>-<PROMPT>.json`
2. Extract fields:
   - `context` (document)
   - `role`
   - `optimized_loqa_questions` (target questions)
3. Build normalized records:
   - `id`, `dataset`, `role`, `document`, `questions`
4. Split each dataset into `train/val` (default ratio `0.8 / 0.2`, deterministic seed).
5. Save per-dataset outputs.
6. Optionally create merged mixes with user-defined sample counts per dataset.
7. Save both:
   - raw JSONL records,
   - SFT-style `prompt/completion` JSONL records.
8. Use a single prompt template by default (no multi-template switching required for now).

## Planned Output Layout

- `Outputs/ft_data/qg/<dataset>/{train,val}.raw.jsonl`
- `Outputs/ft_data/qg/<dataset>/{train,val}.sft.jsonl`
- `Outputs/ft_data/qg/mixes/<mix_name>/{train,val}.raw.jsonl`
- `Outputs/ft_data/qg/mixes/<mix_name>/{train,val}.sft.jsonl`

## Decisions Locked

1. Fine-tuning splits: `train/val` only.
2. Default split ratio: `0.8 / 0.2`.
3. Prompting: one fixed template.

## Still Open (Minor)

1. Default handling for missing train files (skip vs fail).
