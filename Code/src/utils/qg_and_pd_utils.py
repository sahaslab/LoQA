"""
Utility file for question generation and predictions.
"""
import os
import json
import time
import copy
import asyncio
from typing import Any, Dict
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Callable, Optional
from Code.src.utils.io import read_json_file
from Code.src.utils.model_source import get_model
from Code.src.utils.prompts import opt_leakage_check_prompt_template
from Code.src.utils.preprocessing import list_normalization

                            # ------------------------------
                            # Common funnctions used in both question generation and argument extraction
                            # ------------------------------
def read_dataset_split_and_schema(dataset_root: str, dataset_name: str, split_name: str):
    """Read the specific data split and predefined schema for a given dataset.
    First tries to load from HuggingFace (omar-sharif03/{dataset_name}-processed),
    falls back to local files if not available. Schema is always loaded from local files.
    """
    # Loading from HuggingFace first
    try:
        from datasets import load_dataset
        hub_repo_name = f"omar-sharif03/{dataset_name}-processed"
        dataset = load_dataset(hub_repo_name)

        # Getting test split for my experiments (not using dev or train splits for now)
        if split_name in dataset:
            split_data = dataset[split_name].to_list()
        else:
            raise ValueError(f"Split {split_name} not found in dataset {dataset_name}")
        print(f"Successfully loaded {len(split_data)} samples from HuggingFace")
    except Exception as e:
        print(f"Could not load from HuggingFace ({e}), falling back to local files...")
        dataset_dir = os.path.join(dataset_root, dataset_name)
        dataset_split_path = os.path.join(dataset_dir, f"{dataset_name}-processed.json")
        data_dict = read_json_file(dataset_split_path)
        split_data = data_dict[split_name]
        print(f"Loaded {len(split_data)} samples from local file: {dataset_split_path}")
    
    # Schema is always loaded from local files and it is the same for all splits
    dataset_dir = os.path.join(dataset_root, dataset_name)
    schema_path = os.path.join(dataset_dir, f"{dataset_name}-schema.json")
    dataset_schema = read_json_file(schema_path)
    return split_data, dataset_schema

## Reading test split and schema for a given dataset and schema type
def read_data_split(dataset_root: str, dataset_name: str, split_name: str):
    """Read the test split for a given dataset."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    data_split_path = os.path.join(dataset_dir, f"{dataset_name}-{split_name}.json")
    _data = read_json_file(data_split_path)
    # if isinstance(_data, dict):
    #     if split_name in _data:
    #         _data = _data[split_name]
    #     elif split_name == "gold-test" and "test" in _data:
    #         _data = _data["test"]
    #     else:
    #         raise ValueError(f"Split key '{split_name}' not found in {data_split_path}. Keys: {list(_data.keys())}")
    return _data

def read_schema_questions(dataset_root: str, dataset_name: str, question_type: str):
    """Read the test split and schema for a given dataset."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    schema_path = os.path.join(dataset_dir, f"{dataset_name}-{question_type}.json")
    question_schema = read_json_file(schema_path)
    return question_schema

# getting schema-based question for a dataset item based on dataset type (it is different for different datasets based on structue of the data)
def get_schema_question(dataset_name: str, item: Dict[str, Any], dataset_schema: Dict[str, Any]) -> Any:
    """Get schema-based question for a dataset item based on dataset type."""
    if dataset_name == "DiscourseEE":
        event, argument_type, role = item["event"], item["argument-type"], item["role"]
        return dataset_schema[event][argument_type][role]
    if dataset_name in ["PHEE"]:
        argument_type, role = item["argument-type"], item["role"]
        return dataset_schema[argument_type][role]
    if dataset_name == "MACCROBAT":
        event, role = item["event"], item["role"]
        return dataset_schema[event][role]
    # CaseReportBench default
    return dataset_schema[item["role"]]

# if arugment item is none or empty or null, skip the item, otherwise add to valid items
def get_valid_items(dataset_split, num_samples):
    valid_items = []
    sampled = 0
    for item in dataset_split:
        arg_value = item.get("raw-initial-ground-truth", None)
        if arg_value is None or (isinstance(arg_value, (list, str)) and len(arg_value) == 0) or (isinstance(arg_value, list) and len(arg_value) == 1 and arg_value[0] == "null"):
            continue
        if sampled >= num_samples:
            break
        valid_items.append(item)
        sampled += 1
    return valid_items

                            # ------------------------------
                            # getting response from prompt chain and normalizing it to string format
                            # ------------------------------
def normalize_response(response):
    """Normalize response to string format, handling different response types."""
    if isinstance(response, str):
        return response
    elif hasattr(response, 'content'):
        content = response.content
        return content if isinstance(content, str) else str(content)
    elif hasattr(response, 'model_dump'):
        dumped = response.model_dump()
        content = dumped.get('content', str(response))
        return content if isinstance(content, str) else str(content)
    else:
        return str(response)


def format_role_question(question):
    """Normalize role questions to a clean string format for prompt injection.
    When the input is a list (or parses to one), each item is rendered as a bullet line (- item).
    A leading newline is added so the list starts on the line after 'Role Question:' (or similar).
    """
    def _bulleted(lines):
        return "\n" + "\n".join(f"- {q}" for q in lines)

    if isinstance(question, list):
        return _bulleted(str(q) for q in question)
    if isinstance(question, str):
        try:
            parsed = json.loads(question)
            if isinstance(parsed, list):
                return _bulleted(str(q) for q in parsed)
            if isinstance(parsed, dict) and "questions" in parsed:
                return _bulleted(str(q) for q in parsed["questions"])
        except (json.JSONDecodeError, TypeError):
            pass
    return str(question)

def get_response(prompt_chain, input_dict, *, prompt_template=None, print_prompt=False):
    """Get response from prompt chain with retry logic to handle model overload errors.
    If print_prompt is True and prompt_template is provided, prints the formatted prompt before invoking.
    """
    if print_prompt and prompt_template is not None:
        try:
            print("\n" + "=" * 60 + "\n[PROMPT]\n" + "=" * 60)
            print(prompt_template.format(**input_dict))
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"[Warning] Could not format/print prompt: {e}")
    MAX_RETRIES = 10
    RETRY_DELAY = 10  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            response = prompt_chain.invoke(input_dict)
            return normalize_response(response)
        except Exception as e:
            print(f"[Attempt {attempt + 1}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"All {MAX_RETRIES} retries exhausted. Last error: {e}")
                print("Returning empty string as response")
                return ""

async def get_response_async(prompt_chain, input_dict, *, prompt_template=None, print_prompt=False):
    """Async version of get_response using ainvoke for concurrent processing.
    If print_prompt is True and prompt_template is provided, prints the formatted prompt before invoking.
    """
    if print_prompt and prompt_template is not None:
        try:
            print("\n" + "=" * 60 + "\n[PROMPT]\n" + "=" * 60)
            print(prompt_template.format(**input_dict))
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"[Warning] Could not format/print prompt: {e}")
    MAX_RETRIES = 10
    RETRY_DELAY = 10  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            response = await prompt_chain.ainvoke(input_dict)
            return normalize_response(response)
        except Exception as e:
            print(f"[Attempt {attempt + 1}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"All {MAX_RETRIES} retries exhausted. Last error: {e}")
                print("Returning empty string as response")
                return ""

                            # ------------------------------
                            # Question generation async processing
                            # ------------------------------

async def process_qg_item_async(item, question_type, question_chain):
    """Process a single item asynchronously for question generation."""
    role = item.get("role")
    document = item.get("context")
    arg_value = item.get("raw-initial-ground-truth", None)

    # Generating loqa questions asynchronously
    questions = await get_response_async(
        question_chain,
        {"role": role, "document": document, "gt_arguments": arg_value},
    )
    item[f"{question_type}_questions"] = questions
    return item


async def process_qg_items_async(valid_items, question_type, question_chain, dataset_name, batch_size, rows, num_samples):
    """Process items in batches asynchronously for question generation."""
    with tqdm(total=num_samples, desc=f"Processing {dataset_name} (async)", unit="item") as pbar:
        for i in range(0, len(valid_items), batch_size):
            batch = valid_items[i:i + batch_size]
            tasks = [
                process_qg_item_async(item, question_type, question_chain)
                for item in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            rows.extend([copy.deepcopy(item) for item in batch_results])
            pbar.update(len(batch_results))

                            # ------------------------------
                            # Agument extraction (prediction) async processing
                            # ------------------------------
async def process_pd_item_async(item, question_type, arg_pd_chain, *, arg_pd_prompt_template=None, print_prompt=False):
    """Process a single item asynchronously for argument extraction."""
    role = item.get("role")
    document = item.get("context")
    question = item.get(f"{question_type}_questions")
    input_dict = {"role": role, "document": document, "role_question": format_role_question(question)}
    args = await get_response_async(
        arg_pd_chain,
        input_dict,
        prompt_template=arg_pd_prompt_template,
        print_prompt=print_prompt,
    )
    item[f"{question_type}_args"] = args
    return item

async def process_pd_items_async(valid_items, question_type, arg_pd_chain, dataset_name, batch_size, rows, num_samples, *, arg_pd_prompt_template=None, print_prompt=False):
    """Process items in batches asynchronously for argument extraction."""
    with tqdm(total=num_samples, desc=f"Processing {dataset_name} (async)", unit="item") as pbar:
        for i in range(0, len(valid_items), batch_size):
            batch = valid_items[i:i + batch_size]
            tasks = [
                process_pd_item_async(item, question_type, arg_pd_chain, arg_pd_prompt_template=arg_pd_prompt_template, print_prompt=print_prompt)
                for item in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            rows.extend([copy.deepcopy(item) for item in batch_results])
            pbar.update(len(batch_results))

   
   # ------------------------------
   # Leakage check for zero-shot loqa questions
   # ------------------------------

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

def leakage_check_zero_shot_loqa_questions(all_samples):
    """Leakage check for zero-shot loqa questions."""
    
    leakage_check_model = get_model(
        model_origin='dartmouth',
        model_access_string='openai.gpt-oss-120b'
    )
    leakage_check_prompt_chain, leakage_check_prompt_template = opt_leakage_check_prompt_template(
        leakage_check_model,
        prompt_file_path=os.path.join('/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA/Prompts', "lc", "zs-v0.txt"),
    )
    for item in all_samples:
        role = item.get("role")
        current_questions = item.get("loqa_questions")
        gt_args = item.get("raw-initial-ground-truth")
        processed_gt_args = list_normalization(gt_args or [])
        leakage_check_input = {
            "role": role,
            "current_questions": format_arguments_for_prompt(current_questions),
            "gt_arguments": format_arguments_for_prompt(processed_gt_args),
        }
        leakage_check_output = get_response(
            leakage_check_prompt_chain,
            leakage_check_input,
            prompt_template=leakage_check_prompt_template,
            print_prompt=False,
        )
        final_questions, is_leaked = get_leakage_checked_questions(leakage_check_output)
        lc_info = {
            "is_leaked": is_leaked,
            "previous_questions": current_questions,
            "final_questions": final_questions,
        }
        item["lc_info"] = lc_info
        # Only overwrite questions when we got a valid result; otherwise keep current (e.g. after retry exhaustion or malformed JSON)
        if final_questions is not None:
            item["loqa_questions"] = final_questions
    return all_samples