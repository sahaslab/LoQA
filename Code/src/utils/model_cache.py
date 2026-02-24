#!/usr/bin/env python3
"""
Utility functions for caching Hugging Face models locally.
This allows downloading models once and reusing them from local storage.
We can use this script to download and cache models in a specific directory.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your shared Hugging Face cache directory
HF_CACHE_DIR = "/dartfs-hpc/rc/home/j/f006f3j/lab/shared"


def set_hf_cache_dir(cache_dir: str = HF_CACHE_DIR) -> Path:
    """
    Force Hugging Face/Transformers to cache under `cache_dir`.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HUB_CACHE"] = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path)

    return cache_path


def is_model_cached(model_name: str, cache_dir: str = HF_CACHE_DIR) -> bool:
    cache_path = set_hf_cache_dir(cache_dir)
    model_dir = cache_path / f"models--{model_name.replace('/', '--')}"
    return model_dir.exists()

def load_cached_model(
    model_name: str,
    cache_dir: str = HF_CACHE_DIR,
    token: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model/tokenizer using the shared HF cache (downloads if needed).
    """
    cache_path = set_hf_cache_dir(cache_dir)
    if token is None:
        token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_path), token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_path), token=token)
    return tokenizer, model


# Example usage function
def example_usage():
    """
    Example showing how to use the model caching functionality.
    """
    model_name = "Qwen/Qwen3-8B"
    
    # Load model (will download if not cached)
    tokenizer, model = load_cached_model(model_name)
    
    #### Need to change the following parts based on the model we are using ########

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)


if __name__ == "__main__":
    import torch
    example_usage()
