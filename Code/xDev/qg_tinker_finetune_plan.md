# QG Tinker Fine-Tuning Plan (Session Handoff)

## Goal

Fine-tune a QG model with Tinker where:
- Input: `(document, role)` prompt
- Output: JSON question set completion

Data source already prepared:
- `Outputs/ft_data/qg/<dataset_or_mix>/train.sft.jsonl`
- `Outputs/ft_data/qg/<dataset_or_mix>/val.sft.jsonl`

## Decisions Finalized

1. Use `train/val` only for fine-tuning.
2. Use one fixed prompt template (same prompt in data prep and generation).
3. Skip samples longer than `max_seq_len` (do not truncate).
4. Keep only `best` and `final` checkpoints (no periodic checkpoints).
5. Save sampler checkpoints for inference (`best-sampler`, `final-sampler`).

## Current Scripts

1. Trainer:
   - `Code/scripts/8_tinker_qg_finetune.py`
2. Generator:
   - `Code/scripts/9_tinker_qg_generate.py`
3. Dry-run shell wrapper:
   - `Code/scripts/qg_tinker_dryrun.sh`
4. Generation shell wrapper:
   - `Code/scripts/qg_tinker_generate.sh`

## Trainer Behavior (Current)

1. Reads SFT JSONL (`prompt` + `completion`).
2. Converts each example to tokenized Tinker datum.
3. Applies loss mask:
   - prompt tokens weight `0`
   - completion tokens weight `1`
4. Drops examples where tokenized length exceeds `--max-seq-len`.
5. Default `max_seq_len` is `8192`.
6. Tracks validation metric and saves best checkpoint metadata in:
   - `best_checkpoint.json`
7. Saves:
   - `best-state`, `best-sampler`
   - `final-state`, `final-sampler`

## Environment Notes

1. `tinker` package is required in active env.
2. `TINKER_API_KEY` must be exported before running.
3. Python version for Tinker: use Python 3.11 env.

## Run Commands

1. Dry-run (data pipeline check):
   - `bash Code/scripts/qg_tinker_dryrun.sh`
2. Generate from trained run:
   - `bash Code/scripts/qg_tinker_generate.sh`

## Before Next Session

1. Update `RUN_NAME` and document path in `qg_tinker_generate.sh`.
2. If needed, adjust training hparams in `qg_tinker_dryrun.sh`:
   - base model, rank, batch size, learning rate, max examples.
3. Run one full non-dry training job after dry-run validation.
