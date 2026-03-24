#!/usr/bin/env python3
"""
Tinker QG inference — generates questions for each row and saves JSON.
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional


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
    # Remove Qwen3 <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove special tokens like <|im_end|>, <|endoftext|>, etc.
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    # Remove markdown JSON fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def parse_questions(raw_text: str) -> Optional[List[str]]:
    """Try to parse the generated text as a JSON questions list.

    Returns the list of questions if successful, None otherwise.
    """
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

    # Misc
    parser.add_argument("--role-field",     default="role")
    parser.add_argument("--document-field", default="context")
    parser.add_argument("--output-field",   default="ft_questions")
    parser.add_argument("--tinker-api-key-env", default="TINKER_API_KEY")

    args = parser.parse_args()

    # Tinker client
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

    outputs = []
    parse_failures = 0

    for idx, row in enumerate(rows, start=1):
        row_out  = dict(row)
        role     = str(row.get(args.role_field,     "")).strip()
        document = str(row.get(args.document_field, "")).strip()

        if not role or not document:
            row_out["tinker_qg_error"] = "Missing role or document"
            outputs.append(row_out)
            continue

        prompt = DEFAULT_PROMPT_TEMPLATE.format(role=role, document=document)
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        model_input = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

        response = maybe_result(sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ))
        raw_text = tokenizer.decode(response.sequences[0].tokens)

        # Parse the generated JSON
        questions = parse_questions(raw_text)
        if questions is not None:
            row_out[args.output_field] = questions
        else:
            # Save raw text so you can inspect failures
            row_out[args.output_field] = raw_text
            row_out["tinker_qg_parse_error"] = True
            parse_failures += 1

        outputs.append(row_out)

        status = "OK" if questions is not None else "PARSE_FAIL"
        n_qs = len(questions) if questions is not None else "?"
        print(f"\n[{idx}/{len(rows)}] [{status}] serial={row.get('serial-number', idx)}  "
              f"role={role[:30]}  n_questions={n_qs}")
        print(f"[PROMPT]\n{prompt}")
        print(f"[GENERATED]\n{raw_text}")
        print("-" * 60)

    output_file = os.path.join(
        args.output_path,
        args.dataset_name,
        f"{args.question_type}-qg-{args.dataset_name}-{args.split_name}-"
        f"{args.qg_model_name}-{args.qg_prompt_version}.json",
    )
    save_json(outputs, output_file)

    print(f"\n[DONE] Saved {len(outputs)} rows -> {output_file}")
    print(f"  Parsed OK:  {len(outputs) - parse_failures}")
    print(f"  Parse fail: {parse_failures}")
    if parse_failures > 0:
        print(f"  [TIP] Check rows with 'tinker_qg_parse_error': true")


if __name__ == "__main__":
    main()