#!/usr/bin/env python3
"""
Tinker QG inference — generates questions for each row and saves JSON.
Uses async/await with semaphore-bounded concurrency for parallel generation.
"""

import argparse
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm


DEFAULT_PROMPT_TEMPLATE = """\
You are a helpful assistant that generates clear and distinct questions to best extract the ground truth arguments for a given role.
Given a role and a document, output the best set of questions for extracting arguments for that role from the document.
Role:
{role}

Document:
{document}

Return your response strictly in JSON format as:
{{
  "questions": ["your set of questions"]
}}
Do not generate any extra texts or reasonings. Just return the questions in the JSON format.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def maybe_result(value: Any, max_attempts: int = 3, delay: int = 5) -> Any:
    """Resolve a Tinker future synchronously with retries (used for setup calls)."""
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


async def async_maybe_result(value: Any, max_attempts: int = 3, delay: int = 5) -> Any:
    """Resolve a Tinker future asynchronously by offloading .result() to a thread."""
    if not hasattr(value, "result"):
        return value
    for attempt in range(max_attempts):
        try:
            return await asyncio.to_thread(value.result)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"[RETRY] attempt {attempt + 1}/{max_attempts}: {str(e)[:60]}... retrying in {delay}s")
            await asyncio.sleep(delay)


def load_rows(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("Input JSON must be a list.")
    return obj


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_text(text: str) -> str:
    """Remove thinking blocks, special tokens, and markdown fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def parse_questions(raw_text: str) -> Optional[List[str]]:
    """Try to parse the generated text as a JSON questions list."""
    cleaned = clean_text(raw_text)

    # Try direct JSON parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "questions" in obj:
            return obj["questions"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    match = re.search(r'\{[^{}]*"questions"\s*:\s*\[.*?\]\s*\}', cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj.get("questions", [])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def generate_one(
    sem: asyncio.Semaphore,
    pbar: tqdm,
    sampling_client: Any,
    tokenizer: Any,
    sampling_params: Any,
    types: Any,
    orig_idx: int,
    row_out: Dict[str, Any],
    prompt: str,
    role: str,
    total_rows: int,
    output_field: str,
) -> Dict[str, Any]:
    """Generate questions for a single row, bounded by semaphore."""
    async with sem:
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        model_input = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        response = await async_maybe_result(future)
        raw_text = tokenizer.decode(response.sequences[0].tokens)

    # Parse outside the semaphore
    questions = parse_questions(raw_text)
    if questions is not None:
        row_out[output_field] = questions
    else:
        row_out[output_field] = raw_text
        row_out["tinker_qg_parse_error"] = True

    status = "OK" if questions is not None else "PARSE_FAIL"
    n_qs = len(questions) if questions is not None else "?"
    pbar.set_postfix_str(f"{status} role={role[:20]} qs={n_qs}")
    pbar.update(1)

    tqdm.write(f"\n[{orig_idx + 1}/{total_rows}] [{status}] "
               f"serial={row_out.get('serial-number', orig_idx + 1)}  "
               f"role={role[:30]}  n_questions={n_qs}")
    tqdm.write(f"[PROMPT]\n{prompt}")
    tqdm.write(f"[GENERATED]\n{raw_text}")
    tqdm.write("-" * 60)

    return row_out


async def run_inference(args, sampling_client, tokenizer, sampling_params, types, rows) -> None:
    """Main async inference loop."""
    sem = asyncio.Semaphore(args.max_concurrent)
    tasks = []
    skipped = []

    # Count valid rows first for accurate progress bar
    valid_rows = []
    for idx, row in enumerate(rows):
        row_out  = dict(row)
        role     = str(row.get(args.role_field,     "")).strip()
        document = str(row.get(args.document_field, "")).strip()

        if not role or not document:
            row_out["tinker_qg_error"] = "Missing role or document"
            skipped.append((idx, row_out))
            continue
        valid_rows.append((idx, row_out, role, document))

    print(f"[INFO] {len(valid_rows)} rows to generate, {len(skipped)} skipped, "
          f"max_concurrent={args.max_concurrent}")

    pbar = tqdm(total=len(valid_rows), desc="Generating", unit="row")

    for (idx, row_out, role, document) in valid_rows:
        prompt = DEFAULT_PROMPT_TEMPLATE.format(role=role, document=document)
        task = asyncio.create_task(generate_one(
            sem=sem,
            pbar=pbar,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            types=types,
            orig_idx=idx,
            row_out=row_out,
            prompt=prompt,
            role=role,
            total_rows=len(rows),
            output_field=args.output_field,
        ))
        tasks.append((idx, task))

    # Await all tasks
    results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
    pbar.close()

    # Combine results with skipped rows, preserving original order
    all_outputs: List[tuple] = list(skipped)
    parse_failures = 0
    errors = 0

    for (idx, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"[ERROR] Row {idx}: {result}")
            row_out = dict(rows[idx])
            row_out["tinker_qg_error"] = str(result)
            all_outputs.append((idx, row_out))
            errors += 1
        else:
            all_outputs.append((idx, result))
            if result.get("tinker_qg_parse_error"):
                parse_failures += 1

    # Sort by original index
    all_outputs.sort(key=lambda x: x[0])
    outputs = [row_out for (_, row_out) in all_outputs]

    # Save
    output_file = os.path.join(
        args.output_path,
        args.dataset_name,
        f"{args.question_type}-qg-{args.dataset_name}-{args.split_name}-"
        f"{args.qg_model_name}-{args.qg_prompt_version}.json",
    )
    save_json(outputs, output_file)

    print(f"\n[DONE] Saved {len(outputs)} rows -> {output_file}")
    print(f"  Parsed OK:  {len(outputs) - parse_failures - errors}")
    print(f"  Parse fail: {parse_failures}")
    print(f"  Errors:     {errors}")
    if parse_failures > 0:
        print(f"  [TIP] Check rows with 'tinker_qg_parse_error': true")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("Tinker QG inference")

    # Dataset
    parser.add_argument("--dataset-root",      required=True)
    parser.add_argument("--dataset-name",      required=True)
    parser.add_argument("--split-name",        required=True)
    parser.add_argument("--output-path",       required=True)
    parser.add_argument("--question-type",     default="ft")
    parser.add_argument("--qg-model-name",     required=True)
    parser.add_argument("--qg-prompt-version", required=True)

    # Model
    parser.add_argument("--run-dir",    default="", help="Run dir with best_checkpoint.json")
    parser.add_argument("--model-path", default="", help="Direct tinker:// sampler path")

    # Sampling
    parser.add_argument("--max-new-tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top-p",          type=float, default=0.8)
    parser.add_argument("--n-samples",      type=int,   default=-1)
    parser.add_argument("--max-concurrent", type=int,   default=8,
                        help="Max concurrent sample calls (semaphore bound)")

    # Misc
    parser.add_argument("--role-field",     default="role")
    parser.add_argument("--document-field", default="context")
    parser.add_argument("--output-field",   default="ft_questions")
    parser.add_argument("--tinker-api-key-env", default="TINKER_API_KEY")

    args = parser.parse_args()

    # Tinker client (setup is synchronous)
    import tinker        # type: ignore
    from tinker import types  # type: ignore

    api_key = os.getenv(args.tinker_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Set env var '{args.tinker_api_key_env}' before running.")

    service_client = tinker.ServiceClient(api_key=api_key)

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    elif args.run_dir:
        best_path = os.path.join(args.run_dir, "best_checkpoint.json")
        meta = json.load(open(best_path, "r", encoding="utf-8"))
        model_path = meta["save_sampler"]["path"]
        print(f"[INFO] Using sampler: {model_path}")
    else:
        raise ValueError("Provide either --model-path or --run-dir.")

    sampling_client = service_client.create_sampling_client(model_path=model_path)
    tokenizer = maybe_result(sampling_client.get_tokenizer())

    sampling_params = types.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_sequences=["<|im_end|>"],
    )

    # Load data
    input_file = os.path.join(
        args.dataset_root,
        args.dataset_name,
        f"{args.dataset_name}-{args.split_name}.json",
    )
    rows = load_rows(input_file)
    if args.n_samples > 0:
        rows = rows[: args.n_samples]
    print(f"[INFO] Loaded {len(rows)} rows from {input_file}")

    # Run async inference
    asyncio.run(run_inference(args, sampling_client, tokenizer, sampling_params, types, rows))


if __name__ == "__main__":
    main()