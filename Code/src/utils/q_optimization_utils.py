import os, json
import sys
import argparse
import copy
import glob
from typing import Dict, List, Any, Tuple, Callable, Optional
from pprint import pprint
from tqdm import tqdm

# Ensure repository root is on sys.path so "Code" is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.preprocessing import get_process_predictions, list_normalization
from Code.src.utils.qg_and_pd_utils import format_role_question
from Code.src.utils.evaluation_utils import (
    _exact_match_single,
    _compute_all_pairwise_scores,
    _relaxed_match_at_threshold,
    complex_match_two_lists_for_qo
)

def calculating_score_on_iteration(
    iteration_no: int,
    loqa_questions: List[str],
    raw_pred_args: List[str],
    processed_gt_args: List[str],
    role: str,
    context: str,
    judge_model: Any,
    max_concurrency: int = 50,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, float], List[str]]:
    """
    Calculate scores and trace for a single iteration.

    Returns:
        trace_item: Dictionary with all iteration details
        scores: Dictionary with precision, recall, f1
        processed_pred_args: Processed prediction arguments
    """
    pred_args = get_process_predictions(raw_pred_args, role)
    processed_pred_args = list_normalization(pred_args)

    # Exact, relaxed, complex (LLM-as-judge) matching evaluation
    em_matched_pairs, em_remaining_predictions, em_remaining_ground_truth = _exact_match_single(
        processed_pred_args, processed_gt_args, verbose=False
    )

    lst_of_pairs = _compute_all_pairwise_scores(
        em_remaining_predictions,
        em_remaining_ground_truth,
        verbose=False
    )
    rm_matched_pairs, remaining_pd_set, remaining_gt_set = _relaxed_match_at_threshold(
        em_remaining_predictions,
        em_remaining_ground_truth,
        lst_of_pairs,
        threshold=0.85,
        verbose=False
    )

    cm_matched_pairs, remaining_pd_set, remaining_gt_set = complex_match_two_lists_for_qo(
        list(remaining_pd_set), list(remaining_gt_set),
        role, context, judge_model,
        verbose=False
    )

    # Score calculation (set-based to avoid double-counting duplicates)
    pred_unique = set(processed_pred_args)
    gt_unique = set(processed_gt_args)
    remaining_pred_unique = set(remaining_pd_set)
    remaining_gt_unique = set(remaining_gt_set)

    n_pred = len(pred_unique)
    n_gt = len(gt_unique)
    n_matched_pred = n_pred - len(remaining_pred_unique)
    n_matched_gt = n_gt - len(remaining_gt_unique)
    epsilon = 1e-10

    precision = n_matched_pred / max(n_pred, epsilon)
    recall = n_matched_gt / max(n_gt, epsilon)
    f1 = (2 * precision * recall) / max(precision + recall, epsilon)

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_matched_pred": n_matched_pred,
        "n_matched_gt": n_matched_gt,
        "n_pred": n_pred,
        "n_gt": n_gt,
    }

    all_matched_pairs = em_matched_pairs + rm_matched_pairs + cm_matched_pairs

    trace_item = {
        "iteration_no": iteration_no,
        "loqa_questions": loqa_questions,
        "raw_pd_args": raw_pred_args,
        "processed_pd_args": processed_pred_args,
        "em_pairs": em_matched_pairs,
        "rm_pairs": rm_matched_pairs,
        "cm_pairs": cm_matched_pairs,
        "all_matched_pairs": all_matched_pairs,
        "remaining_pd_args": list(remaining_pd_set),
        "remaining_gt_args": list(remaining_gt_set),
        "scores": scores,
    }
    return trace_item, scores, processed_pred_args


def format_arguments_for_prompt(arguments: List[str]) -> str:
    """Format a list of arguments as bulleted markdown for LLM input."""
    if not arguments:
        return "None"
    return "\n".join([f"- {arg}" for arg in arguments])

def get_leakage_checked_questions(leakage_check_output: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Parse leakage check output and return (checked_questions, is_leaked)."""
    try:
        leakage_check_parsed = json.loads(leakage_check_output)
        return (
            leakage_check_parsed.get("final_questions", None),
            leakage_check_parsed.get("is_leaked", None),
        )
    except (json.JSONDecodeError, AttributeError):
        return None, None


def run_refinement_loop_on_single_sample(
    item: Dict,
    initial_loqa_prompt_chain,
    arg_pd_chain,
    refinement_prompt_chain,
    refinement_prompt_template,
    get_response,
    judge_model: Any,
    leakage_check_prompt_chain,
    leakage_check_prompt_template,
    num_iterations: int = 5,
    target_score: float = 1.0,
    maximum_patience: int = 3,
    verbose: bool = False,
    print_prompt: bool = False,
    initial_loqa_prompt_template=None,
    arg_pd_prompt_template=None,
) -> Dict:
    """
    Run the refinement loop on a single item (one sample).

    Reads role, context, and ground truth from `item`; mutates and returns `item`
    with keys: processed_gt_args, trace_history, optimized_loqa_questions,
    optimized_loqa_args, best_trace, best_iteration_number, best_score, total_iterations.
    """
    # Contextual information gathering
    role = item.get("role")
    document = item.get("context")
    context = document if isinstance(document, str) else str(document or "")
    gt_args = item.get("raw-initial-ground-truth")
    processed_gt_args = list_normalization(gt_args or [])
    item["processed_gt_args"] = processed_gt_args

    # Step 0: Generate initial questions (zero-shot with ground truth arguments)
    initial_loqa_input = {
        "role": role,
        "document": document,
        "gt_arguments": format_arguments_for_prompt(gt_args),
    }
    initial_loqa_questions = get_response(
        initial_loqa_prompt_chain,
        initial_loqa_input,
        prompt_template=initial_loqa_prompt_template,
        print_prompt=print_prompt,
    )
    try:
        initial_loqa_parsed = json.loads(initial_loqa_questions)
        initial_loqa_questions = initial_loqa_parsed.get("questions", initial_loqa_questions)
    except (json.JSONDecodeError, AttributeError):
        pass

    trace_history = []
    best_loqa_questions = initial_loqa_questions
    best_loqa_args = None
    best_trace = None
    best_iteration_number = 0
    best_score = 0.0
    patience_counter = 0
    loqa_questions = initial_loqa_questions
    iteration_no = 0

    while iteration_no < num_iterations:
        if verbose:
            print("==========================================")
            print(f"Iteration {iteration_no} of {num_iterations}")
            print("==========================================")

        # Leakage check for current questions
        leakage_check_input = {
            "role": role,
            "current_questions": format_arguments_for_prompt(loqa_questions),
            "gt_arguments": format_arguments_for_prompt(processed_gt_args),
        }
        leakage_check_output = get_response(
            leakage_check_prompt_chain,
            leakage_check_input,
            prompt_template=leakage_check_prompt_template,
            print_prompt=print_prompt,
        )
        final_questions, is_leaked = get_leakage_checked_questions(leakage_check_output)
        lc_info = {
            "is_leaked": is_leaked,
            "previous_questions": loqa_questions,
            "final_questions": final_questions,
        }

        if verbose and not print_prompt:
            print("\n===== LEAKAGE CHECK =====")
            print(leakage_check_prompt_template.format(**leakage_check_input))
            print("+" * 60)
            pprint(lc_info)
            print("=========================\n")

        if final_questions is not None:
            loqa_questions = final_questions

        # Step 1: Predict arguments with current questions
        arg_pd_input = {
            "role": role,
            "document": document,
            "role_question": format_role_question(loqa_questions),
        }
        raw_pred_args = get_response(
            arg_pd_chain,
            arg_pd_input,
            prompt_template=arg_pd_prompt_template,
            print_prompt=print_prompt,
        )

        if verbose and not print_prompt:
            print("\n===== ARGUMENT PREDICTION =====")
            print(arg_pd_prompt_template.format(**arg_pd_input))
            print("+" * 60)
            print(f"Raw Predicted Arguments: {raw_pred_args}")
            print("===============================\n")

        # Step 2 & 3: Calculate scores and compare
        trace, scores, processed_pred_args = calculating_score_on_iteration(
            iteration_no,
            loqa_questions,
            raw_pred_args,
            processed_gt_args,
            role,
            context,
            judge_model,
            verbose=verbose,
        )
        trace["lc_info"] = lc_info
        trace_history.append(trace)

        precision = scores["precision"]
        recall = scores["recall"]
        f1 = scores["f1"]

        # Step 4: Save best and update patience
        # This is because the best score is 0 in the initial iteration, so if we use (>) it will not assign the best_loqa_q and best_loqa_args
        if f1 > best_score or (f1 == best_score and iteration_no == 0): 
            best_score = f1
            best_loqa_questions = loqa_questions
            best_loqa_args = processed_pred_args
            best_trace = trace
            best_iteration_number = iteration_no
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print("Score Summary:")
            pprint(scores)

        # Step 5: Check for early exit conditions
        if best_score >= target_score:
            break
        if patience_counter >= maximum_patience:
            break
        if iteration_no >= num_iterations - 1:
            break

        # Step 6 & 7: Generate feedback and refine questions
        matched_args = [pred for pred, gt in trace["all_matched_pairs"]]
        input_dict = {
            "role": role,
            "document": document,
            "gt_arguments": format_arguments_for_prompt(processed_gt_args),
            "iteration": iteration_no,
            "current_questions": format_arguments_for_prompt(loqa_questions),
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1_score": f"{f1:.4f}",
            "n_matched_gt": scores["n_matched_gt"],
            "n_gt": scores["n_gt"],
            "n_pred": scores["n_pred"],
            "n_matched_pred": scores["n_matched_pred"],
            "matched_arguments": format_arguments_for_prompt(matched_args),
            "missing_arguments": format_arguments_for_prompt(list(trace["remaining_gt_args"])),
            "extra_arguments": format_arguments_for_prompt(list(trace["remaining_pd_args"])),
        }
        try:
            refined_output = get_response(
                refinement_prompt_chain,
                input_dict,
                prompt_template=refinement_prompt_template,
                print_prompt=print_prompt,
            )
            if verbose and not print_prompt:
                print("\n===== REFINEMENT PROMPT =====")
                print(refinement_prompt_template.format(**input_dict))
                print("=============================\n")
            refined_result = json.loads(refined_output)
            assert "questions" in refined_result, "Missing 'questions' key in refinement output"
            assert len(refined_result["questions"]) > 0, "Refined questions list is empty"
            loqa_questions = refined_result["questions"]
        except (json.JSONDecodeError, AssertionError, KeyError):
            pass  # keep current loqa_questions

        # Step 8: Update iteration counter
        iteration_no += 1

    # Step 9: Save final results to item
    item["trace_history"] = trace_history
    item["optimized_loqa_questions"] = best_loqa_questions
    item["optimized_loqa_args"] = best_loqa_args
    item["best_trace"] = best_trace
    item["best_iteration_number"] = best_iteration_number
    item["best_score"] = best_score
    item["total_iterations"] = iteration_no
    return item


def run_refinement_loop_on_dataset(
    sampled_valid_items: List[Dict],
    initial_loqa_prompt_chain,
    arg_pd_chain,
    refinement_prompt_chain,
    refinement_prompt_template,
    get_response,
    judge_model: Any,
    leakage_check_prompt_chain,
    leakage_check_prompt_template,
    num_iterations: int = 5,
    target_score: float = 1.0,
    maximum_patience: int = 3,
    verbose: bool = False,
    print_prompt: bool = False,
    initial_loqa_prompt_template=None,
    arg_pd_prompt_template=None,
    parallel: bool = False,
    max_workers: Optional[int] = 8,
) -> List[Dict]:
    """
    Run refinement loop on each item in the dataset.

    Iterates over `sampled_valid_items` and calls `run_refinement_loop_on_single_sample`
    for each; returns a list of processed items.
    """
    common_kwargs = dict(
        initial_loqa_prompt_chain=initial_loqa_prompt_chain,
        arg_pd_chain=arg_pd_chain,
        refinement_prompt_chain=refinement_prompt_chain,
        refinement_prompt_template=refinement_prompt_template,
        get_response=get_response,
        judge_model=judge_model,
        leakage_check_prompt_chain=leakage_check_prompt_chain,
        leakage_check_prompt_template=leakage_check_prompt_template,
        num_iterations=num_iterations,
        target_score=target_score,
        maximum_patience=maximum_patience,
        verbose=verbose,
        print_prompt=print_prompt,
        initial_loqa_prompt_template=initial_loqa_prompt_template,
        arg_pd_prompt_template=arg_pd_prompt_template,
    )

    # Sequential execution
    if not parallel:
        rows: List[Dict] = []
        for idx, item in enumerate(tqdm(sampled_valid_items, desc="Refinement", unit="item")):
            tqdm.write(
                    f"\n**************************\n"
                    f"Processing item {idx + 1}/{len(sampled_valid_items)} "
                    f"(serial={item.get('serial-number', idx)})\n"
                    f"**************************\n"
                )
            result = run_refinement_loop_on_single_sample(copy.deepcopy(item), **common_kwargs)
            rows.append(result)
        return rows

    # Parallel execution (logs from workers may interleave on stdout)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(8, len(sampled_valid_items))

    print("==========================================")
    print(f"Running refinement loop on {len(sampled_valid_items)} items in parallel with {max_workers} workers")
    print("==========================================")

    rows: List[Optional[Dict]] = [None] * len(sampled_valid_items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(run_refinement_loop_on_single_sample, copy.deepcopy(item), **common_kwargs): idx
            for idx, item in enumerate(sampled_valid_items)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(sampled_valid_items), desc="Refinement (parallel)", unit="item"):
            idx = future_to_idx[future]
            try:
                rows[idx] = future.result()
            except Exception as e:
                rows[idx] = None
    rows = [r for r in rows if r is not None]
    return rows