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
) -> List[str]:
    if device is not None:
        model = model.to(device)
    
    generation_params = {
        "num_return_sequences": num_completions,
        "do_sample": True,
    }
    if "temperature" in generation_args:
        generation_params["temperature"] = generation_args["temperature"]
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
        device=0 if device == "cuda" else -1 if device == "cpu" else device,
    )
    
    all_generated_texts = []
    for prompt in prompts:
        pipeline_outputs = text_generator(
            prompt,
            max_length=max_seq_len,
            **generation_params
        )
        
        generated_texts = [output['generated_text'] for output in pipeline_outputs]
        all_generated_texts.extend(generated_texts)
    
    return all_generated_texts