import random
from typing import List
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline
import torch

from branch_reasoning.prompts import system_prompt

def hf_generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    num_completions: int,
    max_seq_len: int,
    system_prompt: str = system_prompt,  
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
        "temperature": temperature,
    }
    generation_params.update(generation_args)
    if "max_tokens" in generation_params:
        del generation_params["max_tokens"]
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device=0 if device == "cuda" else -1 if device == "cpu" else device,
    )
    
    all_results = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        results = text_generator(
            chat_text,
            max_length=max_seq_len,
            return_full_text=False,
            **generation_params
        )
        
        for result_set in results:
            all_results.append(result_set)
    
    return [result['generated_text'] for result in all_results]