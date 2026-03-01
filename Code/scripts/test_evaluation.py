import os
import sys
import argparse
import copy
from pprint import pprint

# Ensure repository root is on sys.path so "Code" is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.io import read_json_file, save_json_file
from Code.src.utils.preprocessing import get_process_predictions, list_normalization
from Code.src.utils.model_source import get_model
from Code.src.utils.evaluation_utils import (
    doing_exact_match,
    calculating_semantic_score,
    relaxed_match_thresholding,
    doing_complex_matching,
    getting_complex_match_pairs,
    overall_score_on_whole_data,
)

def main(args):
    # Load data
    dataset_name = args.dataset_name
    pd_file_path = f'/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/qo/{dataset_name}/optimized_loqa-{dataset_name}-gold-test-gpt-oss-120b-zs-v0.json'
    pd_data = read_json_file(pd_file_path)

    FIELDS_TO_REMOVE = [
        "trace_history",
        "processed_gt_args",
        "best_trace",
        "optimized_loqa_args",
        "best_iteration_number",
        "best_score",
        "total_iterations",
    ]

    eval_data = []
    for pred in pd_data:
        item = copy.deepcopy(pred)

        gt = pred.get("processed_gt_args") or []
        pd = pred.get("optimized_loqa_args") or []

        item["initial-ground-truth"] = gt if isinstance(gt, (list, tuple)) else []
        item["initial-predictions"] = pd if isinstance(pd, (list, tuple)) else []

        eval_data.append(item)

    cleaned_preds = []
    for ex in eval_data:
        # Make a shallow copy and remove unwanted fields if they exist
        ex_clean = {k: v for k, v in ex.items() if k not in FIELDS_TO_REMOVE}
        cleaned_preds.append(ex_clean)

    prepared_data = cleaned_preds
    print(f"Processing {len(prepared_data)} items")    
    # Step 1: Exact matching
    print("\n=== Step 1: Exact Matching ===")
    results = doing_exact_match(prepared_data, verbose=args.verbose)
    
    # Step 2: Calculate semantic scores
    print("\n=== Step 2: Computing Semantic Scores ===")
    results = calculating_semantic_score(results, verbose=args.verbose)
    
    # Step 3: Apply relaxed matching at different thresholds (optional)
    print("\n=== Step 3: Relaxed Matching ===")
    results = relaxed_match_thresholding(results, args.rm_threshold, verbose=args.verbose)

    # Step 4 (optional): LLM-as-judge complex matching on remaining pairs
    print("\n=== Step 4: Complex Matching (LLM Judge) ===")
    llm_as_judge_model = get_model( ##we are using gpt-oss-120b as the judge model change accordingly when needed
        model_origin='dartmouth',
        model_access_string='openai.gpt-oss-120b',
    )
    results = doing_complex_matching( #this gives us model judgement for every pair
        results,
        llm_as_judge_model,
        args.rm_threshold,
        context_key="context",
        verbose=args.verbose,
    )
    results = getting_complex_match_pairs( #by this function we are keeping only the pairs that the model judged as yes
        results, args.rm_threshold, verbose=args.verbose
    )
    results = overall_score_on_whole_data(results, args.rm_threshold, do_complex_match=True)
    pprint(results)
    save_json_file(results, f'/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/raw/{dataset_name}/test_evaluation_1.json')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test evaluation pipeline.")
    parser.add_argument("--dataset-name", type=str, default="CaseReportBench", help="Dataset name")
    parser.add_argument("--rm-threshold", type=float, default=0.85, help="Relaxed match threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--num-samples", type=int, default=-1, help="Limit number of samples (-1 = all)")
    args = parser.parse_args()

    main(args)