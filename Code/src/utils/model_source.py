"""
Model source utilities for loading different types of language models.
Supports Dartmouth Chat API, OpenAI API, and Hugging Face models.
"""

from langchain_openai import ChatOpenAI
import os
from typing import Optional, Tuple, Union

def get_model(model_origin: str, model_access_string: str, gen_temperature: float = 0.0, 
                gpu_uti: float = 0.9, cache_dir: str = "/dartfs-hpc/rc/home/j/f006f3j/lab/shared",
                reasoning_effort: str = "medium",
                hf_token: Optional[str] = None):
    model = None

    if model_origin == 'dartmouth':  # using Dartmouth Chat API
        print(f"Using Dartmouth Chat API model: {model_access_string}")
        dm_chat_api_key = os.getenv("DARTMOUTH_CHAT_API_KEY")
        if model_access_string in ["openai_responses.gpt-5-mini-2025-08-07"]: # for reasoning models need to set the temparatue to 1. 
            model = ChatOpenAI(
                api_key=dm_chat_api_key,
                temperature=1.0,
                model=model_access_string,
                reasoning_effort=reasoning_effort,
                base_url="https://chat.dartmouth.edu/api"
            )
        else:
            model = ChatOpenAI(
                api_key=dm_chat_api_key,
                temperature=gen_temperature,
                model=model_access_string,
                base_url="https://chat.dartmouth.edu/api"
            )

    elif model_origin == 'openai':  # using OpenAI API
        print(f"Using OpenAI API model: {model_access_string}")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if model_access_string in ["gpt-5-mini-2025-08-07"]:
            model = ChatOpenAI(
                api_key=openai_api_key,
                temperature=1.0,
                model=model_access_string,
                reasoning_effort=reasoning_effort,
            )
        else:
            model = ChatOpenAI(
                api_key=openai_api_key,
                temperature=gen_temperature,
                model=model_access_string,
            )

    elif model_origin == "vllm-serve":  # using Hugging Face with local caching
        """
        command to start the vllm server. I need to change the parameters accodingly.
        vllm serve Qwen/Qwen3-8B \
        --gpu-memory-utilization 0.9 \
        --max-model-len 8192 \
        --download-dir /dartfs-hpc/rc/home/j/f006f3j/lab/shared \
        --port 8000 \
        --trust-remote-code 
        """
        ##the issue with serve is I can not control the thinking mode. It is always enabled by default. 
        print(f"Using vLLM model from server: {model_access_string}")
        model = ChatOpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
            model=model_access_string,
            temperature=gen_temperature ##the issue with t
        )
    elif model_origin == "vllm-local":  # using Hugging Face with local caching
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from langchain_core.runnables import RunnableLambda
        import asyncio

        tokenizer = AutoTokenizer.from_pretrained(model_access_string, cache_dir=cache_dir)
        llm = LLM(
            model=model_access_string,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_uti,
            download_dir=cache_dir,
        )
        sampling_params = SamplingParams(
            temperature=gen_temperature,
            top_p=0.95,
            max_tokens=8192,
        )
        return tokenizer, llm, sampling_params
    else:
        raise ValueError(f"Unsupported model origin: {model_origin}. "
                         f"Supported origins: 'dartmouth', 'openai', 'vllm'")

    if model is None:
        raise RuntimeError(f"Failed to create a model instance for origin '{model_origin}' and access string '{model_access_string}'")

    print(f"Model {model_access_string} loaded successfully")
    return model

