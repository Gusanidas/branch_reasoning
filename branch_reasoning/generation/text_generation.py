import random
from typing import List
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline
import torch

def hf_generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    num_completions: int,
    max_seq_len: int,
    device: str = "cuda",
    generation_args: dict = {},
    temperature: float = 1.0,
) -> List[str]:
    print(f"Inside hf_generate_text")
    if device is not None:
        model = model.to(device)
    print(f"Model moved to device: {device}")
    
    generation_params = {
        "num_return_sequences": num_completions,
        "do_sample": True,
    }
    generation_params["temperature"] = temperature
    if "top_k" in generation_args:
        generation_params["top_k"] = generation_args["top_k"]
    if "top_p" in generation_args:
        generation_params["top_p"] = generation_args["top_p"]
    if "max_new_tokens" in generation_args:
        generation_params["max_new_tokens"] = generation_args["max_new_tokens"]
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #device=0 if device == "cuda" else -1 if device == "cpu" else device,
    )
    generated_texts = text_generator(
        prompts,
        max_length=max_seq_len,
        **generation_params
    )
    
    # Extract just the generated text from each result
    return [result['generated_text'] for prompt_results in generated_texts for result in prompt_results]