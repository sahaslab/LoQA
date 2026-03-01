#!/usr/bin/env python3
"""
- Loads dataset and schema, processes valid samples
- Runs question optimization refinement loop with initial question generation,
  argument prediction, and iterative refinement
"""
import os
import copy
import sys
import argparse
from pprint import pprint

# Ensure repository root is on sys.path so "Code" is importable when run directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.io import save_json_file
from Code.src.utils.model_source import get_model
from Code.src.utils.prompts import (
    question_generation_prompt_template,
    argument_extraction_prompt_template,
    loqa_refinement_prompt_template,
    opt_leakage_check_prompt_template,  
)
from Code.src.utils.qg_and_pd_utils import (
    read_data_split,
    get_valid_items,
    get_response,
)
from Code.src.utils.q_optimization_utils import run_refinement_loop_on_dataset

def main(args):
    verbose = args.verbose

    # ---------------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------------
    dataset_split = read_data_split(args.dataset_root, args.dataset_name, args.split_name)

    if args.num_samples is None or args.num_samples == -1:
        args.num_samples = len(dataset_split)

    valid_items = get_valid_items(dataset_split, args.num_samples)
    num_valid_items = len(valid_items)
    
    print(f"Number of valid items: {num_valid_items}")
    if num_valid_items == 0:
        print("Warning: No valid items found to process!")
        return

    if num_valid_items < args.num_samples:
        print(f"Warning: Requested {args.num_samples} samples but only {num_valid_items} available. Processing {num_valid_items}.")

    # ---------------------------------------------------------------------
    # Initialize refinement and prediction models
    # ---------------------------------------------------------------------
    loqa_model = get_model(
        model_origin=args.qo_model_origin,
        model_access_string=args.qo_model_access_string,
        gen_temperature=args.qo_temperature,
        gpu_uti=args.qo_gpu_util,
        cache_dir=args.cache_dir,
        reasoning_effort=args.qo_reasoning_effort,
    )

    pd_model = get_model(
        model_origin=args.pd_model_origin,
        model_access_string=args.pd_model_access_string,
        gen_temperature=args.pd_temperature,
        gpu_uti=args.pd_gpu_util,
        cache_dir=args.cache_dir,
        reasoning_effort=args.pd_reasoning_effort,
    )

    judge_model_for_complex_matching = get_model(
        model_origin='dartmouth',
        model_access_string='openai.gpt-oss-120b'
    )

    leakage_check_model = get_model(
        model_origin='dartmouth',
        model_access_string='openai.gpt-oss-120b'
    )

    # ---------------------------------------------------------------------
    # Create prompt chains
    # ---------------------------------------------------------------------
    initial_loqa_prompt_chain, initial_loqa_prompt_template = question_generation_prompt_template(
        loqa_model,
        prompt_file_path=os.path.join(args.prompt_dir, "qg", f"{args.initial_qo_prompt_version}.txt"),
    )
    arg_pd_chain, arg_pd_prompt_template = argument_extraction_prompt_template(
        pd_model,
        prompt_file_path=os.path.join(args.prompt_dir, "pd", f"{args.pd_prompt_version}.txt"),
    )
    loqa_opt_prompt_chain, loqa_opt_prompt_template = loqa_refinement_prompt_template(
        loqa_model,
        prompt_file_path=os.path.join(args.prompt_dir, "qo", f"{args.qo_prompt_version}.txt"),
    )

    leakage_check_prompt_chain, lc_prompt_template = opt_leakage_check_prompt_template(
        leakage_check_model,
        prompt_file_path=os.path.join(args.prompt_dir, "lc", "zs-v0.txt"),
    )

    # ---------------------------------------------------------------------
    # Run refinement loop
    # ---------------------------------------------------------------------
    
    print(f"Running refinement on {len(valid_items)} items")
    results = run_refinement_loop_on_dataset(
        sampled_valid_items=valid_items,
        initial_loqa_prompt_chain=initial_loqa_prompt_chain,
        arg_pd_chain=arg_pd_chain,
        refinement_prompt_chain=loqa_opt_prompt_chain,
        refinement_prompt_template=loqa_opt_prompt_template,
        get_response=get_response,
        judge_model=judge_model_for_complex_matching,
        leakage_check_prompt_chain=leakage_check_prompt_chain,
        leakage_check_prompt_template=lc_prompt_template,
        num_iterations=args.num_iterations,
        target_score=args.target_score,
        maximum_patience=args.maximum_patience,
        parallel=args.parallel,
        max_workers=args.max_workers,
        verbose=verbose,
        print_prompt=args.print_prompt,
        initial_loqa_prompt_template=initial_loqa_prompt_template,
        arg_pd_prompt_template=arg_pd_prompt_template,
    )

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
    num_processed = len(results)
    file_name = f"optimized_loqa-{args.dataset_name}-{args.split_name}-{args.qo_model_name}-{args.qo_prompt_version}.json"
    file_path = os.path.join(args.output_path, args.dataset_name, file_name)
    save_json_file(results, file_path)
    if verbose:
        print(f"Processed {num_processed} items. Data saved to: {file_path}")

    # Print results summary (only when verbose)
    for idx, item in enumerate(results):
        print(f"\n{'*'*50}")
        print("Item:", idx, f"(serial={item.get('serial-number', idx)})")
        print("Role:", item.get("role"))
        print("Processed GT Arguments:", item.get("processed_gt_args"))
        print("Optimized LOQA Args:", item.get("optimized_loqa_args"))
        print("Optimized LOQA Questions:", item.get("optimized_loqa_questions"))
        print("Best Iteration Number:", item.get("best_iteration_number"))
        print("Best Score:", item.get("best_score"))
        print("Total Iterations:", item.get("total_iterations"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Question optimization runner")
    parser.add_argument("--dataset-root", default="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Dataset")
    parser.add_argument("--dataset-name", default="CaseReportBench")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--num-samples", type=int, default=None, help="Max valid items to consider; -1 or None = all")

    # Question optimization model
    parser.add_argument("--qo-model-name", default="gpt-oss-120b", help="Model name for filename")
    parser.add_argument("--qo-model-origin", default="dartmouth")
    parser.add_argument("--qo-model-access-string", default="openai.gpt-oss-120b")
    parser.add_argument("--qo-temperature", type=float, default=0.0)
    parser.add_argument("--qo-gpu-util", type=float, default=0.9)
    parser.add_argument("--qo-reasoning-effort", type=str, default="none")
    parser.add_argument("--qo-prompt-version", default="v0")
    parser.add_argument("--initial-qo-prompt-version", default="zs-v0", help="Prompt for initial question generation")

    # Prediction model (for argument extraction)
    parser.add_argument("--pd-model-name", default="gpt-oss-120b")
    parser.add_argument("--pd-model-origin", default="dartmouth")
    parser.add_argument("--pd-model-access-string", default="openai.gpt-oss-120b")
    parser.add_argument("--pd-temperature", type=float, default=0.0)
    parser.add_argument("--pd-gpu-util", type=float, default=0.9)
    parser.add_argument("--pd-reasoning-effort", type=str, default="none")
    parser.add_argument("--pd-prompt-version", default="zs-v0")

    # Refinement loop
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument("--maximum-patience", type=int, default=3)

    # Parallel execution
    parser.add_argument("--parallel", action="store_true", help="Run refinement loop in parallel")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of workers for parallel execution")

    # Paths
    parser.add_argument("--output-path", default="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/qo/")
    parser.add_argument("--prompt-dir", default="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Prompts")
    parser.add_argument("--cache-dir", default="/dartfs-hpc/rc/home/j/f006f3j/lab/shared")
    parser.add_argument("--verbose", action="store_true", help="Print progress and results to stdout")
    parser.add_argument("--print-prompt", action="store_true", help="Print the formatted prompt before each get_response call (initial QG, arg extraction, refinement)")

    args = parser.parse_args()
    main(args)
