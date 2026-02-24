#!/usr/bin/env python3
# data preparation script for DiscourseEE, PHEE, CaseReportBench datasets. Can be used to prepare data for other datasets as well.

import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm


# Ensure repository root is on sys.path so "Code" is importable when run directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.src.utils.io import read_json_file, save_json_file, get_paths
code_path, data_path, output_path = get_paths()

                            # ------------------------------
                            # DiscourseEE data preparation
                            # ------------------------------
def read_data_DiscourseEE(dataset_root: str, dataset_name: str):
    ds_dir = os.path.join(dataset_root, dataset_name)
    train = read_json_file(os.path.join(ds_dir, "final-train.json"))
    dev = read_json_file(os.path.join(ds_dir, "final-dev.json"))
    test = read_json_file(os.path.join(ds_dir, "final-test.json"))

    print(len(train), len(dev), len(test))
    return train, dev, test

def data_prep_DiscourseEE(raw_data, split_name="train"):
    data = []
    unique_number = 1
    for dt in tqdm(raw_data, desc="Preparing data"):
        base_info = {
            "doc_id": dt.get("doc_id"),
            "id": dt.get("id"),
            "event": dt.get("event-label"),
            "post": dt.get("post"),
            "comment": dt.get("comment"),
        }
        base_info["context"] = f"Post: {base_info['post']}\nComment: {base_info['comment']}"

        for arg_type, roles in (dt.get("ground-truth-arguments") or {}).items():
            for role, gt_args in (roles or {}).items():
                only_args = [arg[0] for arg in gt_args]
                implicit_args = []
                scattered_args = []
                for arg in gt_args: # creating separate lists for implicit and scattered arguments
                    if arg[1] == "implicit":
                        implicit_args.append(arg[0])
                    elif arg[1] == "scattered":
                        scattered_args.append(arg[0])
                new_dt = {
                    "serial-number": f"{split_name}-{unique_number}",
                    **base_info,
                    "argument-type": arg_type,
                    "role": role,
                    "raw-initial-ground-truth": only_args,
                    "raw-implicit-arguments": implicit_args,
                    "raw-scattered-arguments": scattered_args,
                    "raw-gt-args-type": gt_args,
                }
                data.append(new_dt)
                unique_number += 1
    return data

                            # ------------------------------
                            # PHEE data preparation
                            # ------------------------------

def read_data_PHEE(dataset_root: str, dataset_name: str):
    ds_dir = os.path.join(dataset_root, dataset_name)
    train = read_json_file(os.path.join(ds_dir, "train.json"))
    dev = read_json_file(os.path.join(ds_dir, "dev.json"))
    test = read_json_file(os.path.join(ds_dir, "test.json"))
    print(len(train), len(dev), len(test))
    return train, dev, test
    
def data_prep_PHEE(raw_data, split_name="train"):
    data = []
    unique_number = 1
    for dt in tqdm(raw_data):
        base_info = {
            'id': dt.get('id'),
            'event': dt.get('event-label'),
            'trigger': dt.get('trigger'),
            'is-multi-event': dt.get('is-multi-event'),
            'context': dt.get('context'),
        }
        for arg_type, roles in dt.get('ground-truth-arguments', {}).items():
            for role, gt_args in roles.items():
                new_dt = {
                    'serial-number': f"{split_name}-{unique_number}",
                    **base_info,
                    'argument-type': arg_type,
                    'role': role,
                    'raw-initial-ground-truth': gt_args,
                }
                data.append(new_dt)
                unique_number += 1
    return data
                            # ------------------------------
                            # CaseReportBench data preparation
                            # ------------------------------

def read_data_CaseReportBench(dataset_root: str, dataset_name: str):
    from datasets import load_dataset
    print("Reading CaseReportBench data from HuggingFace")
    ds = load_dataset("cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark")
    # Convert the HuggingFace dataset to a list of dicts (JSON-serializable)
    # Since CaseReportBench has as small number of samples thinking about every sample as test sample
    test_data = ds["train"].to_list()
    return test_data 

def data_prep_CaseReportBench(raw_data, split_name="test"):
    roles = [k for k in raw_data[0].keys() if k not in ["pmcid", "text"]]
    ROLE_NAME_MAP = {
        "Vitals_Hema": "Vitals-and-Hematology",
        "EENT": "Eyes-Ears-Nose-Throat",
        "Neuro": "Neurology",
        "CVS": "Cardiovascular-System",
        "RESP": "Respiratory-System",
        "GI": "Gastrointestinal-System",
        "GU": "Genitourinary-System",
        "MSK": "Musculoskeletal-System",
        "DERM": "Dermatology",
        "LYMPH": "Lymphatic-System",
        "ENDO": "Endocrinology",
        "Pregnancy": "Pregnancy",
        "Lab_Image": "Laboratory-and-Imaging",
        "History": "Patient-History",
        "Age (at case presentation)": "Age-at-Presentation",
        "Age (of onset)": "Age-of-Onset",
        "Confirmed_Diagnosis(IEM)": "Confirmed-Diagnosis-IEM",
        "IEM_Treatment": "IEM-Treatment",
    }

    data = []
    unique_number = 1
    for dt in raw_data:
        base_info = {
            "pmcid": dt.get("pmcid"),
            "text": dt.get("text"),
            'context': dt.get("text")
        }
        for role in roles:
            new_dt = {
                "serial-number": f"{split_name}-{unique_number}",
                **base_info,
                "role": ROLE_NAME_MAP.get(role),
                "role-name": role,
                "raw-initial-ground-truth": dt.get(role),
            }
            data.append(new_dt)
            unique_number += 1
    return data

def push_to_hub(processed_data_dict, hub_repo_name, private=False):
    """
    Push processed datasets to HuggingFace Hub.
    
    Args:
        processed_data_dict: Dictionary with keys like 'train', 'dev', 'test' and values as lists of dicts
        hub_repo_name: Full repository name (e.g., 'username/dataset-name' or 'org/dataset-name')
        private: Whether the dataset should be private (default: False)
    """
    from datasets import Dataset, DatasetDict
    
    print(f"\n{'='*60}")
    print(f"Pushing dataset to HuggingFace Hub: {hub_repo_name}")
    print(f"{'='*60}\n")
    
    # Convert lists of dicts to HuggingFace Dataset objects
    dataset_dict = {}
    for split_name, data_list in processed_data_dict.items():
        if data_list:  # Only create dataset if there's data
            print(f"Creating Dataset object for '{split_name}' split ({len(data_list)} samples)...")
            dataset_dict[split_name] = Dataset.from_list(data_list)
            print(f"Created '{split_name}' dataset")
    
    # Create DatasetDict
    hf_dataset = DatasetDict(dataset_dict)

    print(f"\nPushing to hub: {hub_repo_name}")
    hf_dataset.push_to_hub(
        repo_id=hub_repo_name,
        private=private,
        commit_message=f"Upload {list(dataset_dict.keys())} splits"
    )
    print(f"\nSuccessfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_name}")
    print(f"\nYou can now load it with:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{hub_repo_name}')")

def main(args):
    dataset_name = args.dataset_name
    output_root = args.output_root
    # dynamically choosing the function based on dataset name
    data_prep_func = globals().get(f"data_prep_{dataset_name}")
    read_data_func = globals().get(f"read_data_{dataset_name}")
    if read_data_func is None:
        raise ValueError(f"No data reading function found for dataset: {dataset_name}")
    if data_prep_func is None:
        raise ValueError(f"No data preparation function found for dataset: {dataset_name}")

    # Saving the dataset locally and pushing to the hub
    if dataset_name == "CaseReportBench":
        # For CaseReportBench, only test split is used
        test = read_data_func(args.dataset_root, dataset_name)
        processed_data_dict = {'test': data_prep_func(test, split_name='test')}
    else:
        train, dev, test = read_data_func(args.dataset_root, dataset_name)
        processed_data_dict = {
            'train': data_prep_func(train, split_name='train'),
            'dev': data_prep_func(dev, split_name='dev'),
            'test': data_prep_func(test, split_name='test')
        }

    # Save processed json locally
    save_json_file(
        processed_data_dict,
        os.path.join(output_root, dataset_name, f"{dataset_name}-processed.json")
    )

    # Optionally push to hub if requested
    if args.push_to_hub:
        if not args.hub_repo_name:
            raise ValueError("--hub-repo-name is required when --push-to-hub is set")
        push_to_hub(
            processed_data_dict,
            hub_repo_name=args.hub_repo_name,
            private=args.private
        )


if __name__ == "__main__":
    # dataset_name = "CaseReportBench"
    parser = ArgumentParser(description="Data preparation script")
    parser.add_argument("--dataset-root", type=str, default=data_path, help="Root path containing the dataset directory (default: ../Dataset from CWD)")
    parser.add_argument("--dataset-name", type=str, default="DiscourseEE", help="Dataset name (default: DiscourseEE)")
    parser.add_argument("--output-root", type=str, default=data_path, help="Root path to write outputs; defaults to dataset directory")
    parser.add_argument("--push-to-hub", action="store_true", help="Push processed dataset to HuggingFace Hub")
    parser.add_argument("--hub-repo-name", type=str, help="HuggingFace Hub repository name (e.g., 'username/dataset-name' or 'org/dataset-name'). Required if --push-to-hub is set.")
    parser.add_argument("--private", action="store_true", help="Make the dataset private on HuggingFace Hub (default: False, dataset will be public)")
    args = parser.parse_args()
    main(args)


# To push to HuggingFace Hub:
# 1. First, login to HuggingFace: huggingface-cli login
#    Or set HF_TOKEN environment variable: export HF_TOKEN=your_token_here
# 2. Run with --push-to-hub flag:
# python scripts/1_data_prep_hf.py --dataset-name DiscourseEE --push-to-hub --hub-repo-name your-username/discourseee-processed
# python scripts/1_data_prep_hf.py --dataset-name PHEE --push-to-hub --hub-repo-name your-username/phee-processed
# python scripts/1_data_prep_hf.py --dataset-name CaseReportBench --push-to-hub --hub-repo-name your-username/casereportbench-processed

# To make dataset private:
# python scripts/1_data_prep_hf.py --dataset-name DiscourseEE --push-to-hub --hub-repo-name your-username/discourseee-processed --private

# After pushing, you can load the dataset from anywhere:
# from datasets import load_dataset
# dataset = load_dataset("your-username/discourseee-processed")