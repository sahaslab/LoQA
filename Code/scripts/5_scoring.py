import os
import sys
import argparse

# Ensure repository root is on sys.path so "Code" is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.io import read_json_file, save_json_file
from Code.src.utils.evaluation_utils import getting_role_wise_scores, overall_score_on_whole_data


def main(args):
    """
    Calculate scores from evaluated predictions.
    Can calculate both role-wise and overall scores.
    """
    # Load evaluated data
    file_name = f"{args.pred_key}-{args.qg_model_name}-{args.qg_prompt_version}-{args.dataset_name}-{args.split_name}-{args.pd_model_name}-{args.pd_prompt_version}.json"
    eval_file = os.path.join(
        args.evaluation_path,
        args.dataset_name,
        file_name
     )
    
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluated file not found: {eval_file}")
    
    print(f"Loading evaluated data from: {eval_file}")
    evaluated_data = read_json_file(eval_file)
    
    print(f"Threshold: {args.rm_threshold}")
    do_complex_match = args.do_complex_match
    if not do_complex_match:
        print("Complex-match scores disabled (--no-complex-match).")
    
    results = {}
    
    # Overall scores (complex match included when evaluated file has after-complex-match-ground-truth and do_complex_match=True)
    if args.overall_scores:
        print(f"\n=== Calculating Overall Scores ===")
        overall_results = overall_score_on_whole_data(
            predictions=evaluated_data,
            threshold=args.rm_threshold,
            do_complex_match=do_complex_match,
        )
        results['overall'] = overall_results
    
    # Role-wise scores
    if args.role_wise_scores:
        unique_roles = list(set(item.get('role', '') for item in evaluated_data if item.get('role')))
        unique_roles.sort()
        print(f"\n=== Calculating Role-wise Scores ===")
        print(f"Found {len(unique_roles)} unique roles: {unique_roles}")
        role_wise_results = getting_role_wise_scores(
            predictions=evaluated_data,
            unique_roles=unique_roles,
            threshold=args.rm_threshold,
            do_complex_match=do_complex_match,
        )
        results['role-wise'] = role_wise_results
    
    # Define output file path for saving scores
    output_file = os.path.join(
        args.scores_path,
        args.dataset_name,
        f"{args.pred_key}-{args.qg_model_name}-{args.qg_prompt_version}-{args.dataset_name}-{args.split_name}-{args.pd_model_name}-{args.pd_prompt_version}.json"
    )
    
    # Save results
    if args.save_scores:
        save_json_file(results, output_file)
        print(f"\nScores saved to: {output_file}")
    
    # Log to wandb if requested
    if args.use_wandb:
        from Code.src.utils.wandb_utils import log_scores
        log_scores(
            results=results,
            file_path=output_file,
            pred_key=args.pred_key,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            qg_model_name=args.qg_model_name,
            qg_prompt_version=args.qg_prompt_version,
            pd_model_name=args.pd_model_name,
            pd_prompt_version=args.pd_prompt_version,
            project=args.wandb_project or "loqa-scores",
            api_key=args.wandb_api_key,
            config={
                "overall_scores": args.overall_scores,
                "role_wise_scores": args.role_wise_scores,
                "rm_threshold": args.rm_threshold,
                "do_complex_match": do_complex_match,
            },
        )

    # Print summary
    if args.verbose:
        print("\n=== Summary ===")
        
        if 'overall' in results:
            print("\nOverall Scores:")
            metrics = results['overall']
            print(f"  Exact Match - P: {metrics['exact-match-precision']:.2f}, "
                f"R: {metrics['exact-match-recall']:.2f}, "
                f"F1: {metrics['exact-match-f1']:.2f}")
            print(f"  Relaxed Match - P: {metrics['relaxed-match-precision']:.2f}, "
                f"R: {metrics['relaxed-match-recall']:.2f}, "
                f"F1: {metrics['relaxed-match-f1']:.2f}")
            if 'complex-match-precision' in metrics:
                print(f"  Complex Match - P: {metrics['complex-match-precision']:.2f}, "
                    f"R: {metrics['complex-match-recall']:.2f}, "
                    f"F1: {metrics['complex-match-f1']:.2f}")
            print(f"  Counts - GT: {metrics['ground-truth-count']}, "
                f"PD: {metrics['prediction-count']}, "
                f"EM: {metrics['exact-match-count']}, "
                f"RM: {metrics['relaxed-match-count']}", end="")
            if 'complex-match-count' in metrics:
                print(f", CM: {metrics['complex-match-count']}")
            else:
                print()

        # print role-wise score make things difficult to inspect
        # if 'role-wise' in results:
        #     print("\nRole-wise Scores:")
        #     for role, metrics in results['role-wise'].items():
        #         print(f"\n  {role}:")
        #         print(f"    Exact Match - P: {metrics['exact-match-precision']:.2f}, "
        #             f"R: {metrics['exact-match-recall']:.2f}, "
        #             f"F1: {metrics['exact-match-f1']:.2f}")
        #         print(f"    Relaxed Match - P: {metrics['relaxed-match-precision']:.2f}, "
        #             f"R: {metrics['relaxed-match-recall']:.2f}, "
        #             f"F1: {metrics['relaxed-match-f1']:.2f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring calculator")
    parser.add_argument("--dataset-name", help="Dataset name")
    parser.add_argument("--split-name", help="Split name")
    parser.add_argument("--pred-key", help="Prediction key")
    parser.add_argument("--qg-model-name", help="Question generation model name")
    parser.add_argument("--qg-prompt-version", help="Question generation prompt version")
    parser.add_argument("--pd-model-name", help="Prediction model name")
    parser.add_argument("--pd-prompt-version", help="Prediction prompt version")
    parser.add_argument("--rm-threshold", type=float, default=0.85, help="Relaxed match threshold used in evaluation")
    parser.add_argument("--overall-scores", action="store_true", help="Calculate overall scores on whole data")
    parser.add_argument("--role-wise-scores", action="store_true", help="Calculate role-wise scores")
    parser.add_argument("--save-scores", action="store_true", help="Save scores to a file")
    parser.add_argument("--use-wandb", action="store_true", help="Log scores to wandb")
    parser.add_argument("--wandb-project", help="Wandb project name")
    parser.add_argument("--wandb-api-key", help="Wandb API key")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--evaluation-path", type=str, default='/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/ev/', help="Path to save evaluation results")
    parser.add_argument("--scores-path", type=str, default='/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/sc/', help="Path to save scores")
    parser.add_argument("--do-complex-match", action="store_true", help="Include complex-match in scores when the evaluated file has complex-match results")
    args = parser.parse_args()
    main(args)

