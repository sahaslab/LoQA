import os
import sys
import argparse
import copy

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
)

def prepare_data_for_evaluation(data, key):
    """
    Prepare data for evaluation by extracting and normalizing predictions.
    
    Args:
        data: List of dictionaries with predictions and ground truth
        key: Either "schema" or "loqa" to choose which predictions to evaluate
    
    Returns:
        List of dictionaries with 'initial-ground-truth' and 'initial-predictions' ready for matching
    """
    prepared_data = []
    gt_key = "raw-initial-ground-truth"
    pred_key = f"{key}_args"

    for item in data:
        new_item = copy.deepcopy(item) 
        # Get ground truth (already a list)
        raw_gt = item.get(gt_key, [])
        # Normalize ground truth
        normalized_gt = list_normalization(raw_gt) if raw_gt else []
        
        # Get predictions (raw string from schema_args)
        raw_pred = item.get(pred_key, "")
        role = item.get("role", "")
        
        # Extract, process and normalize predictions
        processed_pred = get_process_predictions(raw_pred, role)
        normalized_pred = list_normalization(processed_pred)
        
        # Store normalized versions
        new_item['initial-ground-truth'] = normalized_gt
        new_item['initial-predictions'] = normalized_pred
        prepared_data.append(new_item)
    return prepared_data

def main(args):
    # Load data
    pd_file_name = f"{args.question_type}-pd-{args.qg_model_name}-{args.qg_prompt_version}-{args.dataset_name}-{args.split_name}-{args.pd_model_name}-{args.pd_prompt_version}.json"
    pd_file_path = os.path.join(args.pd_path, args.dataset_name, pd_file_name)
    if not os.path.exists(pd_file_path):
        print(f"No predictions generated for {args.question_type} questions for {args.dataset_name} {args.split_name} with model:{args.pd_model_name} and prompt:{args.pd_prompt_version}")
        print(f"Please run the prediction script first")
        return
    try:
        pd_data = read_json_file(pd_file_path)
    except FileNotFoundError:
        print(f"Error reading file: {pd_file_path}")
        return

    # Prepare data: extract and normalize predictions and ground truth
    prepared_data = prepare_data_for_evaluation(pd_data, args.question_type)
    if args.num_samples is not None and args.num_samples != -1 and len(prepared_data) > args.num_samples:
        print(f"Warning: Requested {args.num_samples} samples but only {len(prepared_data)} available. Processing {len(prepared_data)}.")
        prepared_data = prepared_data[:args.num_samples]

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
    if args.do_complex_match:
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

    # Save results
    if args.evaluation_path:
        output_file = os.path.join(
            args.evaluation_path, 
            args.dataset_name,
            f"{args.question_type}-{args.qg_model_name}-{args.qg_prompt_version}-{args.dataset_name}-{args.split_name}-{args.pd_model_name}-{args.pd_prompt_version}.json"
        )
        save_json_file(results, output_file)
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset evaluation runner")
    parser.add_argument("--dataset-name", default="CaseReportBench", help="Dataset name to use")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--question-type", default="loqa", help="Question type to use")
    parser.add_argument("--qg-model-name", help="Question generation model name to use")
    parser.add_argument("--qg-prompt-version", help="Question generation prompt version to use")
    parser.add_argument("--pd-path", type=str, default='/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/pd/', help="Path to read predictions")
    parser.add_argument("--pd-model-name", help="Model name to use")
    parser.add_argument("--pd-prompt-version", type=str, help="Argument extraction prompt version")
    parser.add_argument("--rm-threshold", type=float, default=0.85, help="Relaxed match threshold used in evaluation")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print progress information")
    parser.add_argument("--evaluation-path", type=str, default='/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/ev/', help="Path to save evaluation results")
    # LLM-as-judge complex matching
    parser.add_argument("--do-complex-match", action="store_true", help="Run LLM judge on remaining pairs after relaxed match")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to evaluate")
    args = parser.parse_args()
    main(args)