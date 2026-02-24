"""
Evaluation utilities for exact, relaxed, and complex (LLM-as-judge) matching
of predictions vs ground truth.
"""

import asyncio
import copy
import time
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import threading

# Global variables for caching (to avoid reloading model and recomputing embeddings) and thread safety locking for embeddings cache
_sbert_model = None
_embeddings_cache = {}
_embeddings_lock = threading.Lock()


#----------------------------------------------------------
# ---------------- Exact Matching ----------------
#----------------------------------------------------------
def _exact_match_single(predictions: List[str], ground_truth: List[str], verbose: bool = True) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Perform exact matching between predictions and ground truth for a single item.
    
    Args:
        predictions: List of prediction strings
        ground_truth: List of ground truth strings
        verbose: Whether to print progress
    
    Returns:
        - matched_pairs: List of (prediction, ground_truth) tuples
        - remaining_predictions: Predictions that didn't match
        - remaining_ground_truth: Ground truth that didn't match
    """
    # Use sets for efficient matching
    pred_set = set(predictions)
    gt_set = set(ground_truth)
    
    # Find exact matches (intersection)
    matched_items = pred_set & gt_set
    
    # Convert to pairs
    matched_pairs = []
    for item in matched_items:
        matched_pairs.append((item, item))
        if verbose:
            print(f"{item} || {item}")
    
    # Get remaining items
    remaining_predictions = list(pred_set - matched_items)
    remaining_ground_truth = list(gt_set - matched_items)
    
    return matched_pairs, remaining_predictions, remaining_ground_truth

def doing_exact_match(predictions: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Args:
        predictions: List of dictionaries with 'initial-ground-truth' and 'initial-predictions' keys
        verbose: Whether to print progress
    
    Returns:
        List of dictionaries with exact match results added
    """
    new_pred_dictionary = []
    
    for dt in predictions:
        new_dt = copy.deepcopy(dt)
        
        normalized_actual_labels = dt.get('initial-ground-truth', [])
        normalized_predictions = dt.get('initial-predictions', [])
        
        # Perform exact matching
        em_pair, remaining_pred, remaining_gt = _exact_match_single(
            normalized_predictions, 
            normalized_actual_labels,
            verbose=verbose
        )
        
        # Store results
        new_dt['after-exact-match-ground-truth'] = remaining_gt
        new_dt['after-exact-match-predictions'] = remaining_pred
        new_dt['exact-match-pairs'] = em_pair
        
        new_pred_dictionary.append(new_dt)
    
    return new_pred_dictionary

#----------------------------------------------------------
# ---------------- Relaxed Matching ----------------
#----------------------------------------------------------

def _get_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Get or create global SBERT model instance."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model

def _compute_similarity(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute cosine similarity between two texts using SBERT."""
    global _embeddings_cache
    
    model = _get_sbert_model(model_name)
    
    with _embeddings_lock:
        if text1 not in _embeddings_cache:
            _embeddings_cache[text1] = model.encode(text1, convert_to_tensor=True)
        if text2 not in _embeddings_cache:
            _embeddings_cache[text2] = model.encode(text2, convert_to_tensor=True)
        emb1 = _embeddings_cache[text1]
        emb2 = _embeddings_cache[text2]
    
    cosine_score = util.cos_sim(emb1, emb2)
    return cosine_score.item()

def _compute_all_pairwise_scores(
    predictions: List[str], 
    ground_truth: List[str],
    verbose: bool = True,
    skip_null: bool = True
) -> List[Tuple[Tuple[str, str], float]]:
    """
    Compute pairwise similarity scores between all prediction-ground truth pairs.
    
    Args:
        predictions: List of prediction strings
        ground_truth: List of ground truth strings
        verbose: Whether to print progress
        skip_null: Whether to skip predictions that are 'null'
    
    Returns:
        List of ((prediction, ground_truth), score) tuples, sorted by score descending
    """
    all_pairs = []
    
    for pred in predictions:
        if skip_null and pred == 'null':
            continue
        for gt in ground_truth:
            score = _compute_similarity(pred, gt)
            all_pairs.append(((pred, gt), score))
            # if verbose:
            #     print(f"{pred} || {gt} || {score:.4f}")
    
    # Sort by score descending
    all_pairs.sort(key=lambda x: x[1], reverse=True)
    return all_pairs

def calculating_semantic_score(
    prediction_dictionary: List[Dict[str, Any]], 
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Calculate semantic similarity scores for all pairs after exact matching.
    
    Args:
        prediction_dictionary: List of dictionaries (should have exact match results)
        verbose: Whether to print progress
    
    Returns:
        List of dictionaries with pairwise scores added
    """
    new_pred_dictionary = []
    
    for dt in prediction_dictionary:
        new_dt = copy.deepcopy(dt)
        
        # Get remaining predictions and ground truth after exact match
        normalized_actual_labels = copy.deepcopy(dt.get('after-exact-match-ground-truth', []))
        normalized_predictions = copy.deepcopy(dt.get('after-exact-match-predictions', []))
        
        if verbose:
            print(f"GT: {normalized_actual_labels}")
            print(f"PD: {normalized_predictions}")
        
        # Calculate pairwise matching scores
        lst_of_pairs = _compute_all_pairwise_scores(
            normalized_predictions,
            normalized_actual_labels,
            verbose=verbose
        )
        
        new_dt['relaxed-match-sim-score-all-pairs'] = copy.deepcopy(lst_of_pairs)
        new_pred_dictionary.append(new_dt)
    
    return new_pred_dictionary

def _relaxed_match_at_threshold(
    predictions: List[str],
    ground_truth: List[str],
    pairwise_scores: List[Tuple[Tuple[str, str], float]],
    threshold: float,
    verbose: bool = True
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Perform relaxed matching based on similarity threshold.
    
    Args:
        predictions: Original predictions list
        ground_truth: Original ground truth list
        pairwise_scores: Pre-computed pairwise scores
        threshold: Similarity threshold for matching
        verbose: Whether to print progress
    
    Returns:
        - matched_pairs: List of (prediction, ground_truth) tuples above threshold
        - remaining_predictions: Unmatched predictions
        - remaining_ground_truth: Unmatched ground truth
    """
    # Use sets for efficient lookups
    remaining_pred_set = set(predictions)
    remaining_gt_set = set(ground_truth)
    matched_pairs = []
    
    # Process scores in descending order (greedy matching)
    for (pred, gt), score in pairwise_scores:
        if score < threshold:
            break  # Scores are sorted descending
        
        # Only match if both are still available
        if pred in remaining_pred_set and gt in remaining_gt_set:
            matched_pairs.append((pred, gt))
            remaining_pred_set.remove(pred)
            remaining_gt_set.remove(gt)
            
            if verbose:
                print(f"{pred} || {gt} || {score:.4f}")
    
    return matched_pairs, list(remaining_pred_set), list(remaining_gt_set)

def relaxed_match_thresholding(
    predictions: List[Dict[str, Any]], 
    threshold: float,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Apply relaxed matching at a given similarity threshold.
    Args:
        predictions: List of dictionaries (should have pairwise scores)
        threshold: Similarity threshold for matching
        verbose: Whether to print progress
    
    Returns:
        List of dictionaries with relaxed match results added
    """
    new_pred_dictionary = []
    
    for dt in predictions:
        new_dt = copy.deepcopy(dt)
        
        # Get remaining predictions and ground truth after exact match
        initial_labels = copy.deepcopy(dt.get('after-exact-match-ground-truth', []))
        initial_pred = copy.deepcopy(dt.get('after-exact-match-predictions', []))
        
        # Get pre-computed pairwise scores
        lst_of_pairs = copy.deepcopy(dt.get('relaxed-match-sim-score-all-pairs', []))
        
        # Perform relaxed matching
        rm_pair, remaining_pred, remaining_gt = _relaxed_match_at_threshold(
            initial_pred,
            initial_labels,
            lst_of_pairs,
            threshold,
            verbose=verbose
        )
        
        # Store results
        new_dt[f'after-relaxed-match-{threshold}-ground-truth'] = copy.deepcopy(remaining_gt)
        new_dt[f'after-relaxed-match-{threshold}-predictions'] = copy.deepcopy(remaining_pred)
        new_dt[f'relaxed-match-{threshold}-pairs'] = copy.deepcopy(rm_pair)
        
        new_pred_dictionary.append(new_dt)
    
    return new_pred_dictionary

#----------------------------------------------------------
# ---------------- Complex Matching ----------------
#----------------------------------------------------------
def _get_judge_chain(judge_model: Any, prompt_template=None):
    """Build a reusable LangChain judge chain."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if prompt_template is None:
        prompt_template = PromptTemplate(
            input_variables=["x", "y", "role", "context"],
            template="""Determine if two event arguments refer to the same core entity, value, or claim for the given role.

Match if:
- They convey the same core meaning, even with minor differences in detail, formatting, or specificity.

Context: {context}
Role: {role}
Argument 1: {x}
Argument 2: {y}

Answer "yes" or "no" only."""
        )
    return prompt_template | judge_model | StrOutputParser()

async def _async_judge_one_pair(chain, x: str, y: str, role: str, context: str) -> str:
    """Judge one pair asynchronously, retrying until success."""
    inp = {"x": x, "y": y, "role": role, "context": context}
    while True:
        try:
            try:
                output = await chain.ainvoke(inp)
            except NotImplementedError:
                output = await asyncio.to_thread(chain.invoke, inp)
            return "yes" if "yes" in output.lower() else "no"
        except Exception as e:
            print(e)
            await asyncio.sleep(3)

async def _doing_complex_matching_async(
    prediction_dictionary: List[Dict[str, Any]],
    judge_model: Any,
    threshold: float,
    context_key: str = "context",
    verbose: bool = True,
    max_concurrency: int = 50,
) -> List[Dict[str, Any]]:
    """Collect all pairs, judge them in parallel, group results back."""
    # 1. Collect all (item_idx, pred, gt, role, context) tasks
    tasks = []
    for idx, dt in enumerate(prediction_dictionary):
        remaining_pred = dt.get(f"after-relaxed-match-{threshold}-predictions", [])
        remaining_gt = dt.get(f"after-relaxed-match-{threshold}-ground-truth", [])
        context = dt.get(context_key) or ""
        role = dt.get("role") or ""
        if isinstance(context, dict):
            context = str(context)
        for p in remaining_pred:
            if p == "null":
                continue
            for g in remaining_gt:
                tasks.append((idx, p, g, role, context))

    if not tasks:
        # No pairs to judge; return dicts with empty complex-match-all-pairs
        return [
            {**copy.deepcopy(dt), "complex-match-all-pairs": []}
            for dt in prediction_dictionary
        ]

    chain = _get_judge_chain(judge_model)
    sem = asyncio.Semaphore(max_concurrency)

    if verbose:
        print(f"Dispatching {len(tasks)} judge calls (max_concurrency={max_concurrency}) ...")

    # 2. Run all in parallel (bounded by semaphore)
    async def _run(item_idx, pred, gt, role, ctx):
        async with sem:
            result = await _async_judge_one_pair(chain, pred, gt, role, ctx)
        if verbose:
            print(f"[{item_idx}] {pred} || {gt} || {result}")
        return item_idx, pred, gt, result

    results = await asyncio.gather(*[_run(*t) for t in tasks])

    # 3. Group back per item
    grouped: Dict[int, list] = {}
    for item_idx, pred, gt, result in results:
        grouped.setdefault(item_idx, []).append(((pred, gt), result))

    # 4. Attach to dicts
    new_pred_dictionary = []
    for idx, dt in enumerate(prediction_dictionary):
        new_dt = copy.deepcopy(dt)
        new_dt["complex-match-all-pairs"] = grouped.get(idx, [])
        new_pred_dictionary.append(new_dt)
    return new_pred_dictionary

async def _async_complex_match_two_lists(
    predictions: List[str],
    ground_truth: List[str],
    role: str,
    context: str,
    judge_model: Any,
    max_concurrency: int = 50,
    verbose: bool = True,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Run LLM-as-judge on all pred×gt pairs, then greedy one-to-one match on 'yes' results.
    
    Returns:
        matched_pairs, remaining_predictions, remaining_ground_truth
    """
    if isinstance(context, dict):
        context = str(context)

    # Collect tasks
    tasks = []
    for p in predictions:
        if p == "null":
            continue
        for g in ground_truth:
            tasks.append((p, g))

    if not tasks:
        return [], list(predictions), list(ground_truth)

    chain = _get_judge_chain(judge_model)
    sem = asyncio.Semaphore(max_concurrency)

    if verbose:
        print(f"Dispatching {len(tasks)} judge calls (max_concurrency={max_concurrency}) ...")

    async def _run(pred, gt):
        async with sem:
            result = await _async_judge_one_pair(chain, pred, gt, role, context)
        if verbose:
            print(f"{pred} || {gt} || {result}")
        return pred, gt, result

    results = await asyncio.gather(*[_run(p, g) for p, g in tasks])

    # Greedy one-to-one matching on "yes" results
    remaining_pred_set = set(predictions)
    remaining_gt_set = set(ground_truth)
    matched_pairs = []

    for pred, gt, result in results:
        if result == "yes" and pred in remaining_pred_set and gt in remaining_gt_set:
            matched_pairs.append((pred, gt))
            remaining_pred_set.discard(pred)
            remaining_gt_set.discard(gt)

    return matched_pairs, list(remaining_pred_set), list(remaining_gt_set)

def complex_match_two_lists_for_qo(
    predictions: List[str],
    ground_truth: List[str],
    role: str,
    context: str,
    judge_model: Any,
    max_concurrency: int = 50, ## I am running threadpool executor with max_workers=8, so minimize max_concurrency here. 
    verbose: bool = True,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """Sync wrapper for the async complex matching."""
    return asyncio.run(_async_complex_match_two_lists(
        predictions, ground_truth, role, context,
        judge_model, max_concurrency, verbose
    ))

def doing_complex_matching(
    prediction_dictionary: List[Dict[str, Any]],
    judge_model: Any,
    threshold: float,
    context_key: str = "context",
    verbose: bool = True,
    max_concurrency: int = 100,
) -> List[Dict[str, Any]]:
    """
    Runs judge calls in parallel.
    Args:
        prediction_dictionary: List of dictionaries (should have pairwise scores)
        judge_model: Judge model
        threshold: Similarity threshold for matching
        context_key: Key for context
        verbose: Whether to print progress
        max_concurrency: Maximum concurrency for judge calls
    """
    return asyncio.run(_doing_complex_matching_async(
        prediction_dictionary, judge_model, threshold,
        context_key=context_key, verbose=verbose,
        max_concurrency=max_concurrency,
    ))

def getting_complex_match_pairs(
    predictions: List[Dict[str, Any]],
    threshold: float,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    new_pred_dictionary = []
    for dt in predictions:
        new_dt = copy.deepcopy(dt)
        initial_labels = set(dt.get(f"after-relaxed-match-{threshold}-ground-truth", []))
        initial_pred = set(dt.get(f"after-relaxed-match-{threshold}-predictions", []))
        lst_of_pairs = dt.get(f"complex-match-all-pairs", [])
        cm_pair = []
        for value in lst_of_pairs:
            pred, gt, matching_output = value[0][0], value[0][1], value[1]
            if matching_output == "yes" and pred in initial_pred and gt in initial_labels:
                if verbose:
                    print(f"{pred} || {gt} || {matching_output}")
                cm_pair.append((pred, gt))
                initial_pred.discard(pred)
                initial_labels.discard(gt)
        new_dt["after-complex-match-ground-truth"] = list(initial_labels)
        new_dt["after-complex-match-predictions"] = list(initial_pred)
        new_dt["complex-match-pairs"] = cm_pair
        new_pred_dictionary.append(new_dt)
    return new_pred_dictionary

#----------------------------------------------------------
# ---------------- Role-wise Scores ----------------
#----------------------------------------------------------
def getting_role_wise_scores(
    predictions: List[Dict[str, Any]],
    unique_roles: List[str],
    threshold: float,
    do_complex_match: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate exact, relaxed, and optionally complex-match scores per role.

    Args:
        predictions: List of dictionaries with evaluation results
        unique_roles: List of unique role names to calculate scores for
        threshold: Relaxed match threshold used
        do_complex_match: Whether to perform complex matching

    Returns:
        Dictionary mapping role names to their evaluation metrics (precision, recall, F1, counts).
    """
    print(len(predictions))
    results = {}
    use_cm = do_complex_match and any(
        "after-complex-match-ground-truth" in dt for dt in predictions
    )

    for role in unique_roles:
        gt_cnt, pd_cnt, na_e, np_e, na_r, np_r, na_c, np_c = 0, 0, 0, 0, 0, 0, 0, 0
        em_cnt, rm_cnt, cm_cnt = 0, 0, 0

        for dt in predictions:
            if dt["role"] != role:
                continue

            gt_cnt += len(dt["initial-ground-truth"])
            pd_cnt += len(dt["initial-predictions"])

            na_e += len(dt["initial-ground-truth"]) - len(dt["after-exact-match-ground-truth"])
            np_e += len(dt["initial-predictions"]) - len(dt["after-exact-match-predictions"])

            na_r += len(dt["after-exact-match-ground-truth"]) - len(
                dt[f"after-relaxed-match-{threshold}-ground-truth"]
            )
            np_r += len(dt["after-exact-match-predictions"]) - len(
                dt[f"after-relaxed-match-{threshold}-predictions"]
            )

            if use_cm and "after-complex-match-ground-truth" in dt:
                na_c += len(dt[f"after-relaxed-match-{threshold}-ground-truth"]) - len(
                    dt["after-complex-match-ground-truth"]
                )
                np_c += len(dt[f"after-relaxed-match-{threshold}-predictions"]) - len(
                    dt["after-complex-match-predictions"]
                )
                cm_cnt += len(dt.get("complex-match-pairs", []))

            em_cnt += len(dt["exact-match-pairs"])
            rm_cnt += len(dt[f"relaxed-match-{threshold}-pairs"])

        epsilon = 1e-10
        total_cnt = em_cnt + rm_cnt + cm_cnt
        print(role)
        print(f"GT: {gt_cnt}, PD: {pd_cnt}, EM:{na_e, np_e}, RM:{na_r, np_r}, CM:{na_c, np_c}, EM+RM+CM: {total_cnt}")

        em_precision = np_e / max(pd_cnt, epsilon)
        em_recall = na_e / max(gt_cnt, epsilon)
        em_f1 = (2 * em_precision * em_recall) / max((em_precision + em_recall), epsilon)

        rm_precision = (np_e + np_r) / max(pd_cnt, epsilon)
        rm_recall = (na_e + na_r) / max(gt_cnt, epsilon)
        rm_f1 = (2 * rm_precision * rm_recall) / max((rm_precision + rm_recall), epsilon)

        role_result = {
            "exact-match-precision": round(em_precision * 100, 2),
            "exact-match-recall": round(em_recall * 100, 2),
            "exact-match-f1": round(em_f1 * 100, 2),
            "relaxed-match-precision": round(rm_precision * 100, 2),
            "relaxed-match-recall": round(rm_recall * 100, 2),
            "relaxed-match-f1": round(rm_f1 * 100, 2),
            "ground-truth-count": gt_cnt,
            "prediction-count": pd_cnt,
            "exact-match-count": em_cnt,
            "relaxed-match-count": rm_cnt,
        }

        if use_cm:
            cm_precision = (np_e + np_r + np_c) / max(pd_cnt, epsilon)
            cm_recall = (na_e + na_r + na_c) / max(gt_cnt, epsilon)
            cm_f1 = (2 * cm_precision * cm_recall) / max((cm_precision + cm_recall), epsilon)
            role_result["complex-match-precision"] = round(cm_precision * 100, 2)
            role_result["complex-match-recall"] = round(cm_recall * 100, 2)
            role_result["complex-match-f1"] = round(cm_f1 * 100, 2)
            role_result["complex-match-count"] = cm_cnt

        results[role] = role_result

    return results

#----------------------------------------------------------
# ---------------- Overall Scores ----------------
#----------------------------------------------------------
def overall_score_on_whole_data(
    predictions: List[Dict[str, Any]],
    threshold: float,
    do_complex_match: bool = True,
) -> Dict[str, Any]:
    """
    Calculate exact, relaxed, and optionally complex-match scores over all predictions.
    Complex match is included when predictions contain 'after-complex-match-ground-truth'.

    Args:
        predictions: List of dictionaries with evaluation results
        threshold: Relaxed match threshold used

    Returns:
        Dictionary with overall evaluation metrics (precision, recall, F1, counts).
    """
    print(len(predictions))
    gt_cnt, pd_cnt, na_e, np_e, na_r, np_r, na_c, np_c = 0, 0, 0, 0, 0, 0, 0, 0
    em_cnt, rm_cnt, cm_cnt = 0, 0, 0
    use_cm = do_complex_match and any("after-complex-match-ground-truth" in dt for dt in predictions)

    for dt in predictions:
        gt_cnt += len(dt["initial-ground-truth"])
        pd_cnt += len(dt["initial-predictions"])
        na_e += len(dt["initial-ground-truth"]) - len(dt["after-exact-match-ground-truth"])
        np_e += len(dt["initial-predictions"]) - len(dt["after-exact-match-predictions"])
        na_r += len(dt["after-exact-match-ground-truth"]) - len(
            dt[f"after-relaxed-match-{threshold}-ground-truth"]
        )
        np_r += len(dt["after-exact-match-predictions"]) - len(
            dt[f"after-relaxed-match-{threshold}-predictions"]
        )
        if use_cm and "after-complex-match-ground-truth" in dt:
            na_c += len(dt[f"after-relaxed-match-{threshold}-ground-truth"]) - len(
                dt["after-complex-match-ground-truth"]
            )
            np_c += len(dt[f"after-relaxed-match-{threshold}-predictions"]) - len(
                dt["after-complex-match-predictions"]
            )
            cm_cnt += len(dt.get("complex-match-pairs", []))
        em_cnt += len(dt["exact-match-pairs"])
        rm_cnt += len(dt[f"relaxed-match-{threshold}-pairs"])

    epsilon = 1e-10
    total_cnt = em_cnt + rm_cnt + cm_cnt
    print(f"GT: {gt_cnt}, PD: {pd_cnt}, EM:{na_e, np_e}, RM:{na_r, np_r}, CM:{na_c, np_c}, EM+RM+CM: {total_cnt}")

    em_precision = np_e / max(pd_cnt, epsilon)
    em_recall = na_e / max(gt_cnt, epsilon)
    em_f1 = (2 * em_precision * em_recall) / max((em_precision + em_recall), epsilon)
    rm_precision = (np_e + np_r) / max(pd_cnt, epsilon)
    rm_recall = (na_e + na_r) / max(gt_cnt, epsilon)
    rm_f1 = (2 * rm_precision * rm_recall) / max((rm_precision + rm_recall), epsilon)

    results = {
        "exact-match-precision": round(em_precision * 100, 2),
        "exact-match-recall": round(em_recall * 100, 2),
        "exact-match-f1": round(em_f1 * 100, 2),
        "relaxed-match-precision": round(rm_precision * 100, 2),
        "relaxed-match-recall": round(rm_recall * 100, 2),
        "relaxed-match-f1": round(rm_f1 * 100, 2),
        "ground-truth-count": gt_cnt,
        "prediction-count": pd_cnt,
        "exact-match-count": em_cnt,
        "relaxed-match-count": rm_cnt,
    }

    if use_cm:
        cm_precision = (np_e + np_r + np_c) / max(pd_cnt, epsilon)
        cm_recall = (na_e + na_r + na_c) / max(gt_cnt, epsilon)
        cm_f1 = (2 * cm_precision * cm_recall) / max((cm_precision + cm_recall), epsilon)
        results["complex-match-precision"] = round(cm_precision * 100, 2)
        results["complex-match-recall"] = round(cm_recall * 100, 2)
        results["complex-match-f1"] = round(cm_f1 * 100, 2)
        results["complex-match-count"] = cm_cnt

    return results
