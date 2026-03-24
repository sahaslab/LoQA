#!/usr/bin/env python3
"""
Minimal Tinker LoRA fine-tuning for QG SFT data.

Expected input JSONL rows:
  {"prompt": "...", "completion": "..."}
or
  {"messages": [...]}
"""

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def maybe_result(value: Any, max_attempts: int = 3, delay: int = 5) -> Any:
    if not hasattr(value, "result"):
        return value
    for attempt in range(max_attempts):
        try:
            return value.result()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"[RETRY] attempt {attempt + 1}/{max_attempts}: {str(e)[:60]}... retrying in {delay}s")
            time.sleep(delay)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunked(items: List[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i: i + n]


# ---------------------------------------------------------------------------
# Tokenization  (must match inference)
# ---------------------------------------------------------------------------

def build_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    prompt     = str(row.get("prompt", "")).strip()
    completion = str(row.get("completion", "")).strip()
    if not prompt or not completion:
        raise ValueError("Row needs `messages` or both `prompt` and `completion`.")
    return [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def encode_row(row: Dict[str, Any], tokenizer: Any, types: Any,
               max_seq_len: int) -> Optional[Any]:
    """Encode a single-turn user→assistant row into a Tinker Datum.

    Tokenizes prompt and completion independently then concatenates,
    avoiding BPE boundary mismatches from tokenizing the full sequence
    vs. the prompt separately.
    """
    messages = build_messages(row)

    # --- Single-turn: user prompt + assistant header ---
    prompt_text = (f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
                   f"<|im_start|>assistant\n")

    # --- Completion text (what we train on) ---
    comp_text = f"{messages[1]['content']}<|im_end|>\n"

    # --- Tokenize independently, then concatenate ---
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    comp_ids   = tokenizer.encode(comp_text,   add_special_tokens=False)
    full_tokens = prompt_ids + comp_ids

    if len(full_tokens) < 2 or len(full_tokens) > max_seq_len:
        return None

    # Loss weights: 0 for prompt tokens, 1 for completion tokens
    n_prompt = len(prompt_ids)
    weights  = [0.0] * n_prompt + [1.0] * len(comp_ids)

    # Shift for next-token prediction
    input_tokens   = full_tokens[:-1]
    target_tokens  = full_tokens[1:]
    target_weights = weights[1:]

    if not target_tokens:
        return None

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": target_weights},
    )


def encode_dataset(rows, tokenizer, types, max_seq_len) -> Tuple[List[Any], int]:
    data, skipped = [], 0
    for row in rows:
        try:
            d = encode_row(row, tokenizer, types, max_seq_len)
        except Exception as e:
            print(f"[SKIP] encode_row failed: {e}")
            skipped += 1
            continue
        if d is None:
            skipped += 1
        else:
            data.append(d)
    return data, skipped


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def count_completion_tokens(data: List[Any]) -> int:
    """Count tokens with weight > 0 across a list of Datum objects."""
    total = 0
    for datum in data:
        weights = datum.loss_fn_inputs["weights"]
        # Tinker wraps weights in an internal type with a .data attribute
        raw = weights.data if hasattr(weights, "data") else weights
        total += int(np.sum(np.array(raw) > 0))
    return total


def evaluate(training_client: Any, val_data: List[Any], batch_size: int) -> Optional[float]:
    if not val_data:
        return None
    total_loss = 0.0
    total_tokens = 0
    for batch in chunked(val_data, batch_size):
        out = maybe_result(training_client.forward(batch, "cross_entropy"))
        out_d = out if isinstance(out, dict) else (out.model_dump() if hasattr(out, "model_dump") else {})
        v = (out_d.get("metrics") or {}).get("loss:sum")
        if v is not None:
            total_loss += float(v)
            total_tokens += count_completion_tokens(batch)
    return total_loss / total_tokens if total_tokens > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("Minimal Tinker QG fine-tuning")

    # Data
    parser.add_argument("--train-file",   required=True)
    parser.add_argument("--val-file",     default="")
    parser.add_argument("--output-dir",   default="Outputs/ft_runs/qg_tinker")
    parser.add_argument("--run-name",     default="")

    # Model
    parser.add_argument("--base-model",   default="Qwen/Qwen3-8B")
    parser.add_argument("--rank",         type=int,   default=32)

    # Training
    parser.add_argument("--num-epochs",       type=int,   default=3)
    parser.add_argument("--batch-size",       type=int,   default=16)
    parser.add_argument("--val-batch-size",   type=int,   default=32)
    parser.add_argument("--max-seq-len",      type=int,   default=8192)
    parser.add_argument("--learning-rate",    type=float, default=4.7e-4)
    parser.add_argument("--seed",             type=int,   default=7)

    # Logging / checkpointing
    parser.add_argument("--log-every-steps",  type=int,   default=10)
    parser.add_argument("--eval-every-steps", type=int,   default=50)

    # API
    parser.add_argument("--tinker-api-key-env", default="TINKER_API_KEY")
    parser.add_argument("--wandb-project",       default="loqa-qg-finetune")

    args = parser.parse_args()

    random.seed(args.seed)
    run_name = args.run_name or f"qg-{args.base_model.replace('/', '_')}-r{args.rank}-{int(time.time())}"
    run_dir  = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_json(vars(args), os.path.join(run_dir, "config.json"))

    import wandb
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Load data
    train_rows = read_jsonl(args.train_file)
    val_rows   = read_jsonl(args.val_file) if args.val_file else []
    print(f"[RUN]  {run_name}")
    print(f"[DATA] train={len(train_rows)} val={len(val_rows)}")

    # Tinker setup
    api_key = os.getenv(args.tinker_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Set env var '{args.tinker_api_key_env}' before running.")

    try:
        import tinker        # type: ignore
        from tinker import types  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import tinker.") from e

    try:
        service_client = tinker.ServiceClient(api_key=api_key)
    except TypeError:
        service_client = tinker.ServiceClient()

    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
    )
    tokenizer = maybe_result(training_client.get_tokenizer())

    # Encode
    train_data, train_skip = encode_dataset(train_rows, tokenizer, types, args.max_seq_len)
    val_data,   val_skip   = encode_dataset(val_rows,   tokenizer, types, args.max_seq_len)
    print(f"[DATA] encoded train={len(train_data)} (skipped {train_skip})  val={len(val_data)} (skipped {val_skip})")
    if not train_data:
        raise RuntimeError("No training examples after encoding.")

    save_json({"train": len(train_data), "val": len(val_data),
               "train_skipped": train_skip, "val_skipped": val_skip},
              os.path.join(run_dir, "data_summary.json"))

    adam_params = types.AdamParams(learning_rate=args.learning_rate)
    total_steps = args.num_epochs * math.ceil(len(train_data) / args.batch_size)
    print(f"[TRAIN] epochs={args.num_epochs}  total_steps={total_steps}  lr={args.learning_rate}")

    best_val_loss: Optional[float] = None
    best_step = None
    best_path = os.path.join(run_dir, "best_checkpoint.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    step = 0

    def _save_best(step: int, epoch: int, val_loss: float) -> None:
        """Save best checkpoint (training state + sampler weights)."""
        nonlocal best_val_loss, best_step
        best_val_loss = val_loss
        best_step = step
        state_name   = f"best-state-step{step}"
        sampler_name = f"best-sampler-step{step}"
        state_out   = maybe_result(training_client.save_state(name=state_name))
        sampler_out = maybe_result(training_client.save_weights_for_sampler(name=sampler_name))
        state_d   = state_out   if isinstance(state_out,   dict) else (state_out.model_dump()   if hasattr(state_out,   "model_dump") else {"path": getattr(state_out,   "path", "")})
        sampler_d = sampler_out if isinstance(sampler_out, dict) else (sampler_out.model_dump() if hasattr(sampler_out, "model_dump") else {"path": getattr(sampler_out, "path", "")})
        save_json({
            "best_metric":      best_val_loss,
            "best_metric_mode": "min",
            "best_metric_key":  "auto",
            "best_step":        best_step,
            "best_epoch":       epoch,
            "save_state":       state_d,
            "save_sampler":     sampler_d,
        }, best_path)
        print(f"[BEST] step={step}  val_loss={best_val_loss:.4f}")
        wandb.log({"best_val_loss": best_val_loss}, step=step)

    with open(metrics_path, "w", encoding="utf-8") as mf:
        for epoch in range(1, args.num_epochs + 1):
            random.shuffle(train_data)

            for batch in chunked(train_data, args.batch_size):
                step += 1
                fb = maybe_result(training_client.forward_backward(batch, "cross_entropy"))
                maybe_result(training_client.optim_step(adam_params))

                fb_d = fb if isinstance(fb, dict) else (fb.model_dump() if hasattr(fb, "model_dump") else {})
                raw_loss = (fb_d.get("metrics") or {}).get("loss:sum")
                train_loss = float(raw_loss) if raw_loss is not None else None

                batch_completion_tokens = count_completion_tokens(batch)
                train_loss_normalized = (
                    train_loss / batch_completion_tokens
                    if (train_loss is not None and batch_completion_tokens > 0)
                    else None
                )

                if step % args.log_every_steps == 0:
                    if train_loss_normalized is not None:
                        print(f"[STEP] {step}/{total_steps}  epoch={epoch}  "
                              f"train_loss={train_loss_normalized:.4f}  "
                              f"tokens={batch_completion_tokens}")
                    else:
                        print(f"[STEP] {step}/{total_steps}  epoch={epoch}")
                    wandb.log({
                        "train_loss": train_loss_normalized,
                        "train_loss_sum": train_loss,
                        "batch_completion_tokens": batch_completion_tokens,
                        "epoch": epoch,
                    }, step=step)

                if val_data and step % args.eval_every_steps == 0:
                    val_loss = evaluate(training_client, val_data, args.val_batch_size)
                    if val_loss is not None:
                        print(f"[EVAL] step={step}  val_loss={val_loss:.4f}")
                    else:
                        print(f"[EVAL] step={step}  val_loss=n/a")
                    mf.write(json.dumps({"step": step, "epoch": epoch, "val_loss": val_loss}) + "\n")
                    mf.flush()
                    if val_loss is not None:
                        wandb.log({"val_loss": val_loss}, step=step)
                    if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                        _save_best(step, epoch, val_loss)

        # --- Final eval (in case last step wasn't on eval boundary) ---
        if val_data and (step % args.eval_every_steps != 0):
            val_loss = evaluate(training_client, val_data, args.val_batch_size)
            if val_loss is not None:
                print(f"[EVAL-FINAL] step={step}  val_loss={val_loss:.4f}")
                wandb.log({"val_loss": val_loss}, step=step)
                mf.write(json.dumps({"step": step, "epoch": args.num_epochs,
                                     "val_loss": val_loss, "final": True}) + "\n")
                if best_val_loss is None or val_loss < best_val_loss:
                    _save_best(step, args.num_epochs, val_loss)

        # --- Save final checkpoint unconditionally ---
        final_state   = maybe_result(training_client.save_state(name=f"final-state-step{step}"))
        final_sampler = maybe_result(training_client.save_weights_for_sampler(name=f"final-sampler-step{step}"))
        mf.write(json.dumps({"step": step, "final_state": str(final_state),
                             "final_sampler": str(final_sampler)}) + "\n")

        if best_step is not None and best_step != step:
            print(f"[NOTE] Best checkpoint was at step {best_step} (val_loss={best_val_loss:.4f}), "
                  f"not the final step {step}.")
        print(f"[DONE] Final checkpoint saved. Metrics -> {metrics_path}")

    wandb.finish()


if __name__ == "__main__":
    main()