"""Functions for efficiently loading the models, taking advantage of LoRA."""

import gc
import time
from typing import Any, Optional

import numpy as np
from peft import PeftConfig, PeftModel
from peft.tuners.lora import LoraLayer
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    LlamaTokenizer,
)
from tqdm import tqdm

from superhf.utils import print_memory_utilization

def load_eval_model_and_tokenizer(
    model_path: str,
    prev_model: Optional[torch.nn.Module] = None,
    prev_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    verbose: bool = False,
    revision: str = "main",
    **model_kwargs: Any,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """
    Efficiently load a new model and tokenizer, possibly reusing weights from the base model.

    If prev_model is none, simply load the model from the path.

    Otherwise, this assumes we're loading LoRA weights for a model: Check if the LoRA weights for
    model_path will match the prev_model. If so, it will replace any LoRA adapters on prev_model
    with the new adapters. If not, it will reload a new base model and then add the LoRA adapters.

    A similar process happens with loading the appropriate tokenizer.
    """
    # pylint: disable=protected-access

    assert (prev_model is None and prev_tokenizer is None) or (
        prev_model is not None and prev_tokenizer is not None
    ), "Either both prev_model and prev_tokenizer should be None, or neither should."

    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
        tokenizer_path = base_model_path
    except ValueError:
        # Probably isn't a PEFT model, so just load it from scratch
        peft_config = None
        base_model_path = model_path
        tokenizer_path = model_path

    if (
        prev_model is not None
        and peft_config is not None
        and peft_config.base_model_name_or_path == prev_model.config._name_or_path  # type: ignore
    ):
        # We know we have the right base model, reuse it.
        model = prev_model
        if isinstance(model, PeftModel):
            model = model.get_base_model()
        tokenizer = prev_tokenizer
    else:
        # If we don't have a previous model, or it's different from the one we want to load, reload
        if prev_model is not None:
            # Free memory first
            tqdm.write(f"Freeing memory for {model_path}.")
            print_memory_utilization()
            prev_model.cpu()
            del prev_model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            print_memory_utilization()

        if verbose:
            tqdm.write(f"Loading model and tokenizer from scratch for {model_path}.")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        # Fix for misnamed class in the NLP Cluster's Alpaca tokenizer config
        tokenizer_class = (
            LlamaTokenizer
            if "llama" in tokenizer_path or "alpaca" in tokenizer_path
            else AutoTokenizer
        )
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if verbose:
                print(f"Added pad token to tokenizer for {model_path}.")

    if peft_config is not None:
        assert peft_config.base_model_name_or_path == model.config._name_or_path  # type: ignore

        if verbose:
            tqdm.write(f"Loading PEFT adapters for {model_path}.")
        model = PeftModel.from_pretrained(
            model, model_path, revision=revision, **model_kwargs
        )

    # Set eval mode
    # HACK for peft==0.2.0: manually disable merge_weights. Otherwise, .eval() will error.
    for layer in model.modules():
        if isinstance(layer, LoraLayer):
            layer.merge_weights = False
    model.eval()

    # Set device
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return model, tokenizer


