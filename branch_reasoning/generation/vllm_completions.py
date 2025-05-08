from typing import List, Dict, Any, Union, Optional, Tuple, NamedTuple, Iterator
import html
import wandb
import torch
from transformers import PreTrainedTokenizer
from collections import defaultdict
from dataclasses import dataclass
import itertools
import asyncio

from branch_reasoning.generation.branching import QueueDataset, get_new_branches
from branch_reasoning.generation.vllm_generation import vLLM, vllm_generate_text
from branch_reasoning.generation.completions import (
    Branch, 
    BranchedCompletion, 
    ScoringData, 
    Metadata, 
    PromptCompletion,
    _calculate_batch_parameters,
    _fetch_batch,
    _log_statistics,
    _pack_into_return_datatypes
)


async def _perform_vllm_branching(
    vllm_instance: vLLM,
    tokenizer: PreTrainedTokenizer,
    all_completions: Dict[str, str],
    max_branching_points: int,
    branching_factor: int,
    gen_batch_size: int,
    max_len: int,
    generation_args: Dict[str, Any] = {},
) -> Dict[str, str]:
    """
    Perform branching with vLLM instead of Hugging Face.
    """
    new_branches = get_new_branches(
        all_completions,
        branching_factor=branching_factor,
        max_branching_points=max_branching_points,
    )
    queue_dataset = QueueDataset(
        tokenizer=tokenizer,
        batch_size=gen_batch_size,
        # If the branching point is at the end of the sequence, we skip branching it.
        max_seq_len=max_len - 5,
    )
    queue_dataset.add_sequences(new_branches.items())
    
    while not queue_dataset.is_empty():
        keys, batch = queue_dataset.next_batch()
        
        completions = await vllm_generate_text(
            vllm_instance,
            batch,
            num_completions=1,
            max_seq_len=max_len,
            generation_args=generation_args,
        )
        
        new_completions = {}
        for key, completion in zip(keys, completions):
            new_completions[key] = completion
        
        all_completions.update(new_completions)
        new_branches = get_new_branches(
            new_completions,
            branching_factor=branching_factor,
            max_branching_points=max_branching_points,
        )
        queue_dataset.add_sequences(new_branches.items())
    
    return all_completions


async def vllm_generate_completions(
    vllm_instance: vLLM,
    tokenizer: PreTrainedTokenizer,
    dataset: itertools.cycle,
    total_completions: int,
    completions_per_prompt: int,
    gen_batch_size: int,
    current_iter: int,
    max_len: int,
    wandb_logging: bool = True,
    branch_completions: bool = True,
    branching_factor: int = 2,
    max_branching_points: int = 3,
    generation_args: dict = {},
    temperature: float = 1.0,
) -> List[PromptCompletion]:
    """
    Generate completions using vLLM for the given dataset.
    
    This is a vLLM-based version of the generate_completions function from completions.py.
    It uses the vLLM API instead of Hugging Face for text generation.
    """
    (
        prompt_repetitions,
        prompts_per_batch,
        generation_iter,
        num_completions,
    ) = _calculate_batch_parameters(
        completions_per_prompt, gen_batch_size, total_completions
    )
    
    all_completions = {}
    all_prompts = {}
    all_numbers = {}
    all_targets = {}
    all_tags = {}
    all_solutions = {}
    
    total_keys = 0
    
    for i in range(generation_iter):
        prompts, numbers, targets, tags, solutions = _fetch_batch(
            dataset, prompts_per_batch
        )
        
        for j in range(prompt_repetitions):
            # Use vLLM to generate completions
            completions = await vllm_generate_text(
                vllm_instance,
                prompts,
                num_completions=num_completions,
                max_seq_len=max_len,
                generation_args=generation_args,
                temperature=temperature,
            )
            
            if wandb_logging and j == 0:
                # Log example completion
                example_completion = completions[0]
                example_completion_html = wandb.Html(
                    f"<pre>{html.escape(example_completion)}</pre>"
                )
                example_completion_text = wandb.Html(f"```\n{example_completion}\n```")
            
            for k in range(prompts_per_batch):
                total_keys += 1
                base_key = f"{total_keys}#{current_iter}"
                all_prompts[base_key] = prompts[k]
                all_numbers[base_key] = numbers[k]
                all_targets[base_key] = targets[k]
                all_tags[base_key] = tags[k]
                all_solutions[base_key] = solutions[k]
                
                for kk in range(num_completions):
                    completion_key = f"{base_key}_{kk}_" + "_".join(
                        ["0"] * max_branching_points
                    )
                    all_completions[completion_key] = completions[
                        k * num_completions + kk
                    ]
    
    original_no_branches = len(all_completions)
    if branch_completions:
        all_completions = await _perform_vllm_branching(
            vllm_instance,
            tokenizer,
            all_completions,
            max_branching_points,
            branching_factor,
            gen_batch_size,
            max_len,
            generation_args,
        )
    
    if wandb_logging:
        _log_statistics(
            all_completions,
            original_no_branches,
            current_iter,
            example_completion_html,
            example_completion_text,
        )
    
    # Pack into the return datatypes
    return _pack_into_return_datatypes(
        all_completions,
        all_prompts,
        all_numbers,
        all_targets,
        all_tags,
        all_solutions,
    )


if __name__ == "__main__":
    pass