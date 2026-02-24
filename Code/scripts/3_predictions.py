#!/usr/bin/env python3
"""
 - Loads dataset and schema
 - load role-specific questions and predict arguments
 - Supports multiple model origins via Code.src.utils.model_source.get_model
"""
import os
import copy
import sys
import argparse
import asyncio
import glob
from tqdm import tqdm

# Ensure repository root is on sys.path so "Code" is importable when run directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.io import read_json_file, save_json_file
from Code.src.utils.model_source import get_model
from Code.src.utils.prompts import argument_extraction_prompt_template, vllm_argument_extraction_prompt_template
from Code.src.utils.wandb_utils import log_json_predictions
from Code.src.utils.qg_and_pd_utils import (
    get_valid_items,
    get_response,
    format_role_question,
    process_pd_items_async,
)

def main(args):
    # for agument prediction, we need to get the questions first. 
    # loading the question based on question generation model and used prompt
    question_type = args.question_type
    qg_file_name = f"{question_type}-qg-{args.dataset_name}-{args.split_name}-{args.qg_model_name}-{args.qg_prompt_version}.json"
    qg_file_path = os.path.join(args.qg_output_path, args.dataset_name, qg_file_name)

    if not os.path.exists(qg_file_path):
        print(f"No questions generated for {question_type} questions for {args.dataset_name} {args.split_name} with model: {args.qg_model_name} and prompt: {args.qg_prompt_version}")
        print("Please run the question generation script first")
        return

    try:
        qg_data = read_json_file(qg_file_path)
    except Exception as e:
        print(f"Error reading file: {qg_file_path}\n{e}")
        return
    
    if args.num_samples is None or args.num_samples == -1:
        args.num_samples = len(qg_data)

    valid_items = get_valid_items(qg_data, args.num_samples)
    num_valid_items = len(valid_items)

    rows = []
    # Use async processing only for OpenAI and Dartmouth models and vllm-serve
    if args.use_async and args.pd_model_origin in ["openai", "dartmouth", "vllm-serve"]:
        #loading the argument extraction model for async processing
        pd_model = get_model(
            model_origin=args.pd_model_origin,
            model_access_string=args.pd_model_access_string,
            gen_temperature=args.pd_temperature,
            gpu_uti=args.pd_gpu_util,
            cache_dir=args.cache_dir,
            reasoning_effort=args.reasoning_effort,
            hf_token=args.hf_token,
        )
        arg_pd_chain, pd_prompt_template = argument_extraction_prompt_template(pd_model, prompt_file_path=os.path.join(args.prompt_dir, "pd", f"{args.pd_prompt_version}.txt"))
        asyncio.run(process_pd_items_async(valid_items, question_type, arg_pd_chain, args.dataset_name, args.batch_size, rows, num_valid_items, arg_pd_prompt_template=pd_prompt_template, print_prompt=args.print_prompt))
    elif args.pd_model_origin in ["vllm-local"]:
        # Direct vLLM local processing without LangChain wrapper
        tokenizer, llm, sampling_params = get_model(
            model_origin=args.pd_model_origin,
            model_access_string=args.pd_model_access_string,
            gen_temperature=args.pd_temperature,
            gpu_uti=args.pd_gpu_util,
            cache_dir=args.cache_dir,
            reasoning_effort=args.reasoning_effort,
            hf_token=args.hf_token,
        )
        vllm_arg_pd_template = vllm_argument_extraction_prompt_template(prompt_file_path=os.path.join(args.prompt_dir, "pd", f"{args.pd_prompt_version}.txt"))
        formatted_prompts = []
        for item in valid_items:
            role = item.get("role")
            document = item.get("context")
            question = item.get(f"{question_type}_questions")
            inputs = {"role": role, "document": document, "role_question": format_role_question(question)}
            formatted_prompts.append(vllm_arg_pd_template.format(**inputs))

        texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False, #make is false to disable thinking mode
                )
                for prompt in formatted_prompts
            ]
            
        with tqdm(total=num_valid_items, desc=f"Processing {args.dataset_name} (vllm-local) and {args.split_name}", unit="item") as pbar:
            outputs = llm.generate(texts, sampling_params)
            for item, output in zip(valid_items, outputs):
                item[f"{question_type}_args"] = output.outputs[0].text
                rows.append(copy.deepcopy(item))
                pbar.update(1)
    else:
        # Fall back to sync processing for other model origins
        #loading the argument extraction model for sync processing vllm models will note work here
        #it will not 
        pd_model = get_model(
            model_origin=args.pd_model_origin,
            model_access_string=args.pd_model_access_string,
            gen_temperature=args.pd_temperature,
            gpu_uti=args.pd_gpu_util,
            cache_dir=args.cache_dir,
            reasoning_effort=args.reasoning_effort,
            hf_token=args.hf_token,
        )
        arg_pd_chain, pd_prompt_template = argument_extraction_prompt_template(pd_model, prompt_file_path=os.path.join(args.prompt_dir, "pd", f"{args.pd_prompt_version}.txt"))
        with tqdm(total=num_valid_items, desc=f"Processing {args.dataset_name} (sync) and {args.split_name}", unit="item") as pbar:
            for item in valid_items:
                role = item.get("role")
                document = item.get("context")
                question = item.get(f"{question_type}_questions")  # getting the questions that were previously generated
                input_dict = {"role": role, "document": document, "role_question": format_role_question(question)}
                response_args = get_response(
                    arg_pd_chain,
                    input_dict,
                    prompt_template=pd_prompt_template if args.print_prompt else None,
                    print_prompt=args.print_prompt,
                )
                # Extend the item dictionary with schema arguments, and loqa_args
                item[f"{question_type}_args"] = response_args
                rows.append(copy.deepcopy(item))
                pbar.update(1)

    num_processed = len(rows)
    file_name = f"{question_type}-pd-{args.qg_model_name}-{args.qg_prompt_version}-{args.dataset_name}-{args.split_name}-{args.pd_model_name}-{args.pd_prompt_version}.json"
    file_path = os.path.join(args.pd_output_path, args.dataset_name, file_name)
    save_json_file(rows, file_path)
    print(f"Processed {num_processed} items. Data saved to: {file_path}")

    # Log to wandb if requested and logging final predictions
    if args.use_wandb:
        log_json_predictions(
            rows=rows,
            file_path=file_path,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            qg_model_name=args.qg_model_name,
            qg_prompt_version=args.qg_prompt_version,
            pd_model_name=args.pd_model_name,
            pd_prompt_version=args.pd_prompt_version,
            num_samples=num_processed,
            project=args.wandb_project or "loqa-predictions",
            api_key=args.wandb_api_key,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset predictions runner")
    parser.add_argument("--dataset-name", default="CaseReportBench", help="Dataset name to use")
    parser.add_argument("--question-type", default="schema", help="Question type to use, values usually are schema, loqa")
    parser.add_argument("--split-name", default="test", choices=["train", "test", "dev", "gold-test"])
    parser.add_argument("--qg-model-name", help="Question generation model name to use")
    parser.add_argument("--qg-prompt-version", help="Question generation prompt version to use")
    parser.add_argument("--pd-model-origin", default="dartmouth", help="Argument extraction model origin to use")
    parser.add_argument("--pd-model-access-string", help="Argument extraction model access string to get the model from the model origin")
    parser.add_argument("--pd-model-name", help="Model name to use")
    parser.add_argument("--pd-temperature", type=float, default=0.0)
    parser.add_argument("--pd-gpu-util", type=float, default=0.9)
    parser.add_argument("--cache-dir", default="/dartfs-hpc/rc/home/j/f006f3j/lab/shared")
    parser.add_argument("--reasoning-effort", type=str, default="medium", help="Reasoning effort to use for argument extraction")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--qg-output-path", type=str, default='/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/qg/', help="Optional path to save/load json files")
    parser.add_argument("--pd-output-path", type=str, default='/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/pd/', help="Optional path to save/load json files")
    parser.add_argument("--use-async", action="store_true", help="Use async processing for OpenAI/Dartmouth models (faster for API calls)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of items to process concurrently when using async mode")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for accessing gated models (can also be set via HF_TOKEN env var)")
    parser.add_argument("--use-wandb", action="store_true", help="Log results to Weights & Biases (requires wandb to be installed)")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name (default: 'loqa-predictions')")
    parser.add_argument("--wandb-api-key", type=str, default=None, help="Wandb API key for authentication (can also be set via WANDB_API_KEY env var)")
    parser.add_argument("--prompt-dir", type=str, default="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Prompts", help="Prompt directory")
    parser.add_argument("--pd-prompt-version", type=str, default="zs-v0", help="Argument extraction prompt version")
    parser.add_argument("--print-prompt", action="store_true", help="Print the formatted prompt before each get_response call (argument extraction)")
    args = parser.parse_args()
    main(args)


