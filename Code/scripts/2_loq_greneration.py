#!/usr/bin/env python3
"""
 - Loads dataset and schema, and process valid samples
 - Generates refined role-specific list of questions through the selected models
 - Supports multiple model origins via Code.src.utils.model_source.get_model
"""
import os
import copy
import sys
import argparse
import asyncio
from tqdm import tqdm

# Ensure repository root is on sys.path so "Code" is importable when run directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.io import save_json_file
from Code.src.utils.model_source import get_model
from Code.src.utils.prompts import (
    question_generation_prompt_template,
    dynamic_question_generation_prompt_template,
    vllm_question_generation_prompt_template,
) 
from Code.src.utils.qg_and_pd_utils import (
    read_data_split,
    read_schema_questions,
    get_valid_items,
    get_schema_question,
    get_response,
    process_qg_items_async,
    leakage_check_zero_shot_loqa_questions,
)

def main(args):
    dataset_split = read_data_split(args.dataset_root, args.dataset_name, args.split_name)
    qg_prompt_path = os.path.join(args.prompt_dir, "qg", f"{args.qg_prompt_version}.txt")

    if args.num_samples is None or args.num_samples == -1:
        args.num_samples = len(dataset_split)

    # removing null/none/empty ground truth arguments to get valid items to process
    valid_items = get_valid_items(dataset_split, args.num_samples)
    num_valid_items = len(valid_items)
    print(f"Number of valid items: {num_valid_items}")
    
    if num_valid_items == 0:
        print("Warning: No valid items found to process!")
        return
    
    if num_valid_items < args.num_samples:
        print(f"Warning: Requested {args.num_samples} samples but only {num_valid_items} valid items available. Processing {num_valid_items} items.")
    
    rows = []
    question_type = args.question_type

    if question_type in ["schema", "cot-schema"]:
        question_schema = read_schema_questions(args.dataset_root, args.dataset_name, question_type)
        with tqdm(total=num_valid_items, desc=f"Processing {args.dataset_name} and {args.split_name}", unit="item") as pbar:
            for item in valid_items:
                schema_question = get_schema_question(args.dataset_name, item, question_schema)
                item[f"{question_type}_questions"] = schema_question
                rows.append(copy.deepcopy(item))
                pbar.update(1)   
    elif question_type in ["loqa", "dynamicQ"]:
        # Use async processing for OpenAI, Dartmouth, and vllm-serve models
        if args.use_async and args.qg_model_origin in ["openai", "dartmouth", "vllm-serve", "google"]:
            qg_model = get_model(
                model_origin=args.qg_model_origin,
                model_access_string=args.qg_model_access_string,
                gen_temperature=args.qg_temperature,
                gpu_uti=args.qg_gpu_util,
                cache_dir=args.cache_dir,
                reasoning_effort=args.reasoning_effort,
                hf_token=args.hf_token,
            )
            if question_type =="loqa":
                question_chain, _ = question_generation_prompt_template(qg_model, qg_prompt_path)
            if question_type == "dynamicQ":
                dq_prompt_path = os.path.join(args.prompt_dir, "dq", "zs-v0.txt")
                question_chain, _ = dynamic_question_generation_prompt_template(qg_model, dq_prompt_path)
            asyncio.run(process_qg_items_async(valid_items, question_type, question_chain, args.dataset_name, args.batch_size, rows, num_valid_items))

        elif args.qg_model_origin == "vllm-local": # dynamicQ questions are not supported for vllm-local models for now if needed can be added
            tokenizer, llm, sampling_params = get_model(
                model_origin=args.qg_model_origin,
                model_access_string=args.qg_model_access_string,
                gen_temperature=args.qg_temperature,
                gpu_uti=args.qg_gpu_util,
                cache_dir=args.cache_dir,
                reasoning_effort=args.reasoning_effort,
                hf_token=args.hf_token,
            )
            vllm_qg_template = vllm_question_generation_prompt_template(qg_prompt_path)
            formatted_prompts = []
            for item in valid_items:
                role = item.get("role")
                document = item.get("context")
                arg_value = item.get("raw-initial-ground-truth")
                inputs = {"role": role, "document": document, "gt_arguments": arg_value}
                formatted_prompts.append(vllm_qg_template.format(**inputs))
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in formatted_prompts
            ]
            with tqdm(total=num_valid_items, desc=f"Processing {args.dataset_name} (vllm-local) and {args.split_name}", unit="item") as pbar:
                outputs = llm.generate(texts, sampling_params)
                for item, output in zip(valid_items, outputs):
                    item[f"{question_type}_questions"] = output.outputs[0].text
                    rows.append(copy.deepcopy(item))
                    pbar.update(1)
        else:
            # Fall back to sync processing for other model origins
            # Note: vllm-local models will not work here (handled separately above)
            qg_model = get_model(
                model_origin=args.qg_model_origin,
                model_access_string=args.qg_model_access_string,
                gen_temperature=args.qg_temperature,
                gpu_uti=args.qg_gpu_util,
                cache_dir=args.cache_dir,
                reasoning_effort=args.reasoning_effort,
                hf_token=args.hf_token,
            )
            question_chain, _ = question_generation_prompt_template(qg_model, qg_prompt_path)   
            with tqdm(total=num_valid_items, desc=f"Processing {args.dataset_name} and {args.split_name}", unit="item") as pbar:
                for item in valid_items:
                    role = item.get("role")
                    document = item.get("context")
                    arg_value = item.get("raw-initial-ground-truth")
                    questions = get_response(
                        question_chain,
                        {"role": role, "document": document, "gt_arguments": arg_value},
                    )
                    item[f"{question_type}_questions"] = questions
                    rows.append(copy.deepcopy(item))
                    pbar.update(1)

    # Use actual number of processed items in filename
    num_processed = len(rows)
    if question_type == "loqa":
        rows = leakage_check_zero_shot_loqa_questions(rows)
    file_name = f"{question_type}-qg-{args.dataset_name}-{args.split_name}-{args.qg_model_name}-{args.qg_prompt_version}.json"
    file_path = os.path.join(args.output_path, args.dataset_name, file_name)
    save_json_file(rows, file_path)
    print(f"Processed {num_processed} items. Data saved to: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Question generation runner")
    parser.add_argument("--dataset-root", default=os.path.join("/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Dataset"))
    parser.add_argument("--dataset-name", default="CaseReportBench")
    parser.add_argument("--question-type", default="schema", help="Question type to use, values usually are schema, loqa")
    parser.add_argument("--split-name", default="test", help="Split name to use, values usually are train, test, dev")
    parser.add_argument("--qg-model-name", default="gpt-oss-120b", help="Model name for filename (short name without dots/slashes)")
    parser.add_argument("--qg-model-origin", default="dartmouth", help="Model origin to use, values usually are dartmouth, openai, huggingface, vllm, etc.")
    parser.add_argument("--qg-model-access-string", default="openai.gpt-oss-120b", help="Model access string to use. Check carefully which model you are using")
    parser.add_argument("--qg-temperature", type=float, default=0.0, help="Temperature to use for question generation")
    parser.add_argument("--qg-gpu-util", type=float, default=0.9, help="GPU utilization to use for question generation with hf and vllm models")
    parser.add_argument("--cache-dir", default="/dartfs-hpc/rc/home/j/f006f3j/lab/shared", help="Cache directory to use for storing models")
    parser.add_argument("--reasoning-effort", type=str, default="medium", help="Reasoning effort to use for question generation")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for accessing gated models (can also be set via HF_TOKEN env var)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process, if -1 or None, all samples will be processed")
    parser.add_argument("--output-path", type=str, default='/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Outputs/raw/qg/', help="Optional path to save json files")
    parser.add_argument("--use-async", action="store_true", help="Use async processing for OpenAI/Dartmouth models (faster for API calls)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of items to process concurrently when using async mode")
    parser.add_argument("--qg-prompt-version", default="zs-v0", help="Prompt version to use for question generation")
    parser.add_argument("--prompt-dir", default="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Prompts", help="Directory to use for storing prompts")
    args = parser.parse_args()
    main(args)