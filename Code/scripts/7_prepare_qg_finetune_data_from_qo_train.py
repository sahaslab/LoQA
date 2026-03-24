#!/usr/bin/env python3
"""
Prepare question-generator fine-tuning datasets from QO train split only.

Source:
  Outputs/qo/<dataset>/optimized_loqa-<dataset>-train-<model>-<prompt>.json

Output:
  Outputs/ft_data/qg/<dataset>/{train,val}.raw.jsonl
  Outputs/ft_data/qg/<dataset>/{train,val}.sft.jsonl
  Outputs/ft_data/qg/mixes/<mix_name>/{train,val}.raw.jsonl
  Outputs/ft_data/qg/mixes/<mix_name>/{train,val}.sft.jsonl
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path so "Code" is importable when run directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.io import read_json_file, save_json_file


DEFAULT_PROMPT_TEMPLATE = """You are a helpful assistant that generates clear and distinct questions to best extract the ground truth arguments for a given role.
Given a role and a document, output the best set of questions for extracting arguments for that role from the document.
Role:
{role}

Document:
{document}

Return your response strictly in JSON format as:
{{
  "questions": ["your set of questions"]
}}"""


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_questions(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        return cleaned or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                cleaned = [str(x).strip() for x in parsed if str(x).strip()]
                return cleaned or None
            if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                cleaned = [str(x).strip() for x in parsed["questions"] if str(x).strip()]
                return cleaned or None
        except json.JSONDecodeError:
            pass
        lines = [ln.strip("-* ").strip() for ln in text.splitlines() if ln.strip()]
        lines = [x for x in lines if x]
        return lines or None
    return None


def parse_dataset_caps(specs: List[str]) -> Dict[str, int]:
    caps: Dict[str, int] = {}
    for item in specs:
        if "=" not in item:
            raise ValueError(f"Invalid cap spec: {item}. Expected DATASET=INT.")
        dataset, n = item.split("=", 1)
        caps[dataset.strip()] = int(n.strip())
    return caps


def parse_mix_specs(specs: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Example:
      mix_small:PHEE=1000,DiscourseEE=1000
      mix_balanced:PHEE=500,CaseReportBench=500
    """
    mixes: Dict[str, Dict[str, int]] = {}
    for item in specs:
        if ":" not in item:
            raise ValueError(f"Invalid mix spec: {item}. Expected MIX:DATASET=INT,...")
        mix_name, body = item.split(":", 1)
        parts = [x.strip() for x in body.split(",") if x.strip()]
        mixes[mix_name.strip()] = parse_dataset_caps(parts)
    return mixes


def split_train_val(rows: List[Dict[str, Any]], train_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    n = len(rows)
    n_train = int(n * train_ratio)
    return {"train": rows[:n_train], "val": rows[n_train:]}


def sample_rows(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    out = rows[:]
    rng.shuffle(out)
    if n < 0 or n >= len(out):
        return out
    return out[:n]


def build_prompt(prompt_template: str, role: str, document: str) -> str:
    return prompt_template.format(role=role, document=document)


def build_completion(questions: List[str]) -> str:
    return json.dumps({"questions": questions}, ensure_ascii=False)


def to_records(
    dataset: str,
    source_rows: List[Dict[str, Any]],
    role_field: str,
    document_field: str,
    question_field: str,
    gt_arguments_field: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "source_rows": len(source_rows),
        "kept": 0,
        "skip_missing_role_or_document": 0,
        "skip_missing_question_field": 0,
        "skip_empty_questions": 0,
    }
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(source_rows, start=1):
        role = str(row.get(role_field, "")).strip()
        document = str(row.get(document_field, "")).strip()
        if not role or not document:
            stats["skip_missing_role_or_document"] += 1
            continue
        if question_field not in row:
            stats["skip_missing_question_field"] += 1
            continue
        questions = normalize_questions(row.get(question_field))
        if not questions:
            stats["skip_empty_questions"] += 1
            continue

        serial = str(row.get("serial-number", "")).strip()
        rec_id = f"{dataset}:{serial}" if serial else f"{dataset}:{i}"
        out.append(
            {
                "id": rec_id,
                "dataset": dataset,
                "role": role,
                "document": document,
                "questions": questions,
                "ground_truth_arguments": row.get(gt_arguments_field),
            }
        )
        stats["kept"] += 1
    return out, stats


def write_outputs(
    out_dir: str,
    split_rows: Dict[str, List[Dict[str, Any]]],
    prompt_template: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split_name, rows in split_rows.items():
        raw_rows: List[Dict[str, Any]] = []
        sft_rows: List[Dict[str, Any]] = []
        for row in rows:
            raw_rows.append(row)
            sft_rows.append(
                {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "split": split_name,
                    "role": row["role"],
                    "prompt": build_prompt(prompt_template, role=row["role"], document=row["document"]),
                    "completion": build_completion(row["questions"]),
                    "ground_truth_arguments": row.get("ground_truth_arguments"),
                }
            )
        write_jsonl(os.path.join(out_dir, f"{split_name}.raw.jsonl"), raw_rows)
        write_jsonl(os.path.join(out_dir, f"{split_name}.sft.jsonl"), sft_rows)
        counts[split_name] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser("Prepare QG finetuning data from QO train files")
    parser.add_argument("--source-root", default="Outputs/qo")
    parser.add_argument("--output-root", default="Outputs/ft_data/qg")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CaseReportBench", "DiscourseEE", "MACCROBAT", "PHEE"],
    )
    parser.add_argument("--model-name", default="gpt-oss-120b")
    parser.add_argument("--prompt-version", default="zs-v0")
    parser.add_argument(
        "--source-file-template",
        default="optimized_loqa-{dataset}-train-{model_name}-{prompt_version}.json",
    )
    parser.add_argument("--role-field", default="role")
    parser.add_argument("--document-field", default="context")
    parser.add_argument("--question-field", default="optimized_loqa_questions")
    parser.add_argument("--gt-arguments-field", default="raw-initial-ground-truth")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-missing-train-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-samples-per-dataset",
        nargs="*",
        default=[],
        help="Caps like PHEE=5000 DiscourseEE=2000; use -1 for all.",
    )
    parser.add_argument(
        "--mix-specs",
        nargs="*",
        default=[],
        help="Mix specs like mixA:PHEE=2000,DiscourseEE=1000",
    )
    parser.add_argument("--prompt-template-file", default="")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0,1).")

    prompt_template = DEFAULT_PROMPT_TEMPLATE
    if args.prompt_template_file:
        with open(args.prompt_template_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

    caps = parse_dataset_caps(args.max_samples_per_dataset)
    mix_specs = parse_mix_specs(args.mix_specs)
    rng = random.Random(args.seed)

    dataset_pools: Dict[str, List[Dict[str, Any]]] = {}
    per_dataset_meta: Dict[str, Any] = {}

    for dataset in args.datasets:
        filename = args.source_file_template.format(
            dataset=dataset,
            model_name=args.model_name,
            prompt_version=args.prompt_version,
        )
        src_path = os.path.join(args.source_root, dataset, filename)
        if not os.path.exists(src_path):
            msg = f"Missing train source file: {src_path}"
            if args.skip_missing_train_files:
                print(f"[WARN] {msg}")
                continue
            raise FileNotFoundError(msg)

        source_rows = read_json_file(src_path)
        if not isinstance(source_rows, list):
            raise ValueError(f"Expected list JSON in {src_path}")

        pool, stats = to_records(
            dataset=dataset,
            source_rows=source_rows,
            role_field=args.role_field,
            document_field=args.document_field,
            question_field=args.question_field,
            gt_arguments_field=args.gt_arguments_field,
        )
        pool = sample_rows(pool, caps.get(dataset, -1), rng)
        rng.shuffle(pool)
        split_rows = split_train_val(pool, args.train_ratio)

        out_dir = os.path.join(args.output_root, dataset)
        counts = write_outputs(out_dir, split_rows, prompt_template)
        dataset_pools[dataset] = pool
        per_dataset_meta[dataset] = {
            "source_path": src_path,
            "stats": stats,
            "cap": caps.get(dataset, -1),
            "counts": counts,
        }
        print(f"[WRITE] {dataset}: train={counts['train']} val={counts['val']} -> {out_dir}")

    mixes_meta: Dict[str, Any] = {}
    for mix_name, requirements in mix_specs.items():
        merged: List[Dict[str, Any]] = []
        usage: Dict[str, int] = {}
        for dataset, n in requirements.items():
            if dataset not in dataset_pools:
                print(f"[WARN] mix={mix_name}: dataset '{dataset}' unavailable, skipped.")
                continue
            picked = sample_rows(dataset_pools[dataset], n, rng)
            merged.extend(picked)
            usage[dataset] = len(picked)

        rng.shuffle(merged)
        split_rows = split_train_val(merged, args.train_ratio)
        mix_dir = os.path.join(args.output_root, "mixes", mix_name)
        counts = write_outputs(mix_dir, split_rows, prompt_template)
        mixes_meta[mix_name] = {"usage": usage, "counts": counts}
        print(f"[WRITE] mix={mix_name}: train={counts['train']} val={counts['val']} -> {mix_dir}")

    meta = {
        "source_root": args.source_root,
        "output_root": args.output_root,
        "datasets_requested": args.datasets,
        "datasets_loaded": sorted(dataset_pools.keys()),
        "model_name": args.model_name,
        "prompt_version": args.prompt_version,
        "source_file_template": args.source_file_template,
        "role_field": args.role_field,
        "document_field": args.document_field,
        "question_field": args.question_field,
        "gt_arguments_field": args.gt_arguments_field,
        "train_ratio": args.train_ratio,
        "val_ratio": 1.0 - args.train_ratio,
        "seed": args.seed,
        "prompt_template_file": args.prompt_template_file,
        "per_dataset_caps": caps,
        "per_dataset": per_dataset_meta,
        "mixes": mixes_meta,
    }
    meta_path = os.path.join(args.output_root, "build_meta.json")
    save_json_file(meta, meta_path)
    print(f"[OK] Meta saved -> {meta_path}")


if __name__ == "__main__":
    main()
