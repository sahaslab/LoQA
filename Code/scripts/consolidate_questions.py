"""
Consolidate schema, loqa, and optimized_loqa questions for the dev set
of each dataset into a single JSON file for comparative analysis.
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
QG_DIR = BASE_DIR / "Outputs" / "qg"
OUTPUT_DIR = BASE_DIR / "Outputs" / "analysis"

DATASETS = ["CaseReportBench", "DiscourseEE", "MACCROBAT", "PHEE"]
MODEL_TAG = "gpt-oss-120b-zs-v0"
SPLIT = "dev"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_lookup(records, key_fields=("serial-number", "role")):
    lookup = {}
    for rec in records:
        key = tuple(rec[k] for k in key_fields)
        lookup[key] = rec
    return lookup


def consolidate_dataset(dataset):
    qg_path = QG_DIR / dataset

    schema_file = qg_path / f"schema-qg-{dataset}-{SPLIT}-{MODEL_TAG}.json"
    loqa_file = qg_path / f"loqa-qg-{dataset}-{SPLIT}-{MODEL_TAG}.json"
    opt_file = qg_path / f"optimized_loqa-qg-{dataset}-{SPLIT}-{MODEL_TAG}.json"

    for f in [schema_file, loqa_file, opt_file]:
        if not f.exists():
            print(f"  [SKIP] Missing: {f}")
            return None

    schema_data = load_json(schema_file)
    loqa_data = load_json(loqa_file)
    opt_data = load_json(opt_file)

    print(f"  schema: {len(schema_data)} | loqa: {len(loqa_data)} | optimized: {len(opt_data)}")

    loqa_lookup = build_lookup(loqa_data)
    opt_lookup = build_lookup(opt_data)

    consolidated = []
    matched = 0
    for rec in schema_data:
        key = (rec["serial-number"], rec["role"])

        entry = {k: v for k, v in rec.items() if k != "schema_questions"}

        schema_q = rec.get("schema_questions", "")
        if isinstance(schema_q, str):
            schema_q = [schema_q]
        entry["schema_questions"] = schema_q

        loqa_rec = loqa_lookup.get(key)
        if loqa_rec:
            lq = loqa_rec.get("loqa_questions", [])
            entry["loqa_questions"] = lq if isinstance(lq, list) else [lq]
        else:
            entry["loqa_questions"] = []

        opt_rec = opt_lookup.get(key)
        if opt_rec:
            oq = opt_rec.get("optimized_loqa_questions", [])
            entry["optimized_loqa_questions"] = oq if isinstance(oq, list) else [oq]
        else:
            entry["optimized_loqa_questions"] = []

        if loqa_rec and opt_rec:
            matched += 1

        consolidated.append(entry)

    print(f"  Total: {len(consolidated)} | Fully matched: {matched}")
    return consolidated


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print(f"{'='*60}")

        result = consolidate_dataset(dataset)
        if result is None:
            continue

        out_path = OUTPUT_DIR / f"{dataset}-dev-questions-consolidated.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Written: {out_path}")

        print(f"\n  Sample (first record):")
        sample = result[0]
        print(f"    serial-number: {sample['serial-number']}")
        print(f"    role:          {sample['role']}")
        print(f"    schema_q:      {sample['schema_questions']}")
        print(f"    loqa_q:        {sample['loqa_questions']}")
        print(f"    optimized_q:   {sample['optimized_loqa_questions']}")


if __name__ == "__main__":
    main()
