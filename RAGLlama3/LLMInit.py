# -*- coding: utf-8 -*-
# LLMInit.py — initialize_LLM() returns a HuggingFaceLLM (Phi-3) for LlamaIndex 0.10.x

from __future__ import annotations
import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN = os.getenv("HF_TOKEN", "ADD Your hugging face token")  # replace or set env var

def _quiet() -> None:
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if HF_TOKEN:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def initialize_LLM(
    model_id: str = MODEL_ID,
    max_new_tokens: int = 300,
    temperature: float = 0.1,
    context_window: int = 4096,
) -> HuggingFaceLLM:
    """
    Build a LlamaIndex-compatible LLM by names so the wrapper loads a proper model with .config.
    Keep tokenizer == model to avoid fallback mismatches.
    """
    _quiet()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    llm = HuggingFaceLLM(
        model_name=model_id,
        tokenizer_name=model_id,
        model_kwargs={"trust_remote_code": True, "torch_dtype": torch_dtype},
        tokenizer_kwargs={"trust_remote_code": True},
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs={"temperature": temperature},
    )
    return llm
