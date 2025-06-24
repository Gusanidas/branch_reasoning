import random
import json
import os
from typing import List, Optional
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
    assistant_messages: List[str] = None,
    cache_file: Optional[str] = None,
) -> List[str]:
    print(f"Inside hf_generate_text")
    
    # Check if we should use cached completions
    expected_num_completions = len(prompts) * num_completions
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_completions = json.load(f)
            if len(cached_completions) == expected_num_completions:
                print(f"Using cached completions from {cache_file}")
                return cached_completions
            else:
                print(f"Cache file exists but has {len(cached_completions)} completions, expected {expected_num_completions}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error reading cache file {cache_file}: {e}")
    
    if device is not None:
        model = model.to(device)
    print(f"Model moved to device: {device}")
    print(f"In hf generate, generation_args: {generation_args}")
    
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
    if assistant_messages is None:
        assistant_messages = [None] * len(prompts)
    
    # Prepare all chat texts in batch
    batch_texts = []
    for prompt, assistant_message in zip(prompts, assistant_messages):
        print(f"In hf generate, prompt: {prompt[-5:]}, max_seq_len: {max_seq_len}")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Add assistant message if provided for continuation
        extra_args = {}
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})
            extra_args["continue_final_message"] = True
        else:
            extra_args["add_generation_prompt"] = True
        
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **extra_args
        )
        batch_texts.append(chat_text)
    
    # Process all prompts in a single batch
    results = text_generator(
        batch_texts,
        max_length=max_seq_len,
        return_full_text=False,
        max_new_tokens=None,
        truncation=True,
        batch_size=len(prompts),  # Set batch size to process all prompts at once
        **generation_params
    )
    
    # Flatten results since we get num_completions per prompt
    for result_set in results:
        if isinstance(result_set, list):
            all_results.extend(result_set)
        else:
            all_results.append(result_set)
    
    generated_completions = [result['generated_text'] for result in all_results]
    
    # Save to cache file if specified
    if cache_file:
        try:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(generated_completions, f, indent=2)
            print(f"Saved {len(generated_completions)} completions to cache file {cache_file}")
        except Exception as e:
            print(f"Error saving to cache file {cache_file}: {e}")
    
    return generated_completions