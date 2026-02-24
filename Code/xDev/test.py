#!/usr/bin/env python3
"""
Debug script to compare prompts between optimization, sync prediction, and async prediction.
Run on one or more items to see exact prompt differences.
"""
import os
import sys
import json
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

load_dotenv("/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/.env")

from Code.src.utils.io import read_json_file
from Code.src.utils.qg_and_pd_utils import format_role_question
from Code.src.utils.prompts import argument_extraction_prompt_template
from Code.src.utils.model_source import get_model

# ---- Configuration (adjust these) ----
DATASET_NAME = "PHEE"
SPLIT_NAME = "gold-test"
QUESTION_TYPE = "optimized_loqa"
QG_MODEL_NAME = "gpt-oss-120b"
QG_PROMPT_VERSION = "zs-v0"
PD_PROMPT_VERSION = "zs-v0"
PROMPT_DIR = "/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Prompts"
QG_OUTPUT_PATH = "/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/qg/"

# Which items to test:
# - ITEM_INDICES: list of indices, e.g. [0, 5, 20]
# - ITEM_RANGE: (start, end) inclusive range, e.g. (0, 9) for indices 0..9
# - NUM_ITEMS: use first N items (only if both ITEM_INDICES and ITEM_RANGE are None)
ITEM_INDICES = None       # e.g. [0, 5, 20] or None
ITEM_RANGE = (0, 20)       # e.g. (0, 9) for indices 0 through 9 inclusive, or None
NUM_ITEMS = None          # e.g. 5 for first 5 items (ignored if ITEM_INDICES or ITEM_RANGE set)
SHOW_DIAGNOSTIC = True    # print type/diagnostic for role_question per item

# ---- Load data ----
qg_file_name = f"{QUESTION_TYPE}-qg-{DATASET_NAME}-{SPLIT_NAME}-{QG_MODEL_NAME}-{QG_PROMPT_VERSION}.json"
qg_file_path = os.path.join(QG_OUTPUT_PATH, DATASET_NAME, qg_file_name)
qg_data = read_json_file(qg_file_path)

# Resolve which indices to run
if ITEM_INDICES is not None:
    indices = list(ITEM_INDICES)
elif ITEM_RANGE is not None:
    start, end = ITEM_RANGE
    indices = list(range(start, end + 1))  # end inclusive
else:
    n = NUM_ITEMS if NUM_ITEMS is not None else min(5, len(qg_data))
    indices = list(range(n))

# ---- Build the prompt template once ----
pd_model = get_model(
    model_origin="dartmouth",
    model_access_string="openai.gpt-oss-120b",
)
_, prompt_template = argument_extraction_prompt_template(
    pd_model,
    prompt_file_path=os.path.join(PROMPT_DIR, "pd", f"{PD_PROMPT_VERSION}.txt"),
)

print(f"Testing {len(indices)} item(s): indices {indices}")
print("=" * 80)

for idx in indices:
    if idx >= len(qg_data):
        print(f"[Index {idx}] Skipped (data has only {len(qg_data)} items)")
        continue

    item = qg_data[idx]
    role = item.get("role")
    document = item.get("context")
    question = item.get(f"{QUESTION_TYPE}_questions")
    serial = item.get("serial-number", idx)

    print(f"\n{'#' * 80}")
    print(f"# Item index: {idx}  (serial-number: {serial})")
    print(f"{'#' * 80}")
    print(f"Role: {role}")
    print(f"Question type: {type(question).__name__}")
    if SHOW_DIAGNOSTIC:
        if isinstance(question, (list, dict)):
            print(f"Question value:\n{json.dumps(question, indent=2)}")
        else:
            print(f"Question value (str, len={len(question) if question else 0}): {repr(question)[:200]}...")

    formatted_prompt = prompt_template.format(
        role=role,
        document=document,
        role_question=format_role_question(question),
    )

    print("\n----- FORMATTED PROMPT -----")
    print(formatted_prompt)
    print("----- END PROMPT -----")

    if SHOW_DIAGNOSTIC:
        print("\n--- DIAGNOSTIC ---")
        if isinstance(question, list):
            print("  question is a list -> format_role_question outputs bullet list")
        elif isinstance(question, str):
            try:
                parsed = json.loads(question)
                if isinstance(parsed, list):
                    print("  question is JSON string (list)")
                elif isinstance(parsed, dict) and "questions" in parsed:
                    print("  question is JSON string (dict with 'questions')")
                else:
                    print("  question is JSON string (other)")
            except (json.JSONDecodeError, TypeError):
                print("  question is plain string")
        else:
            print(f"  question type: {type(question)}")
    print("=" * 80)