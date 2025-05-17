from typing import List, Dict, Any, Union, Optional, Tuple, NamedTuple, Iterator
import time
import html
import wandb
import torch
import os
from transformers import PreTrainedTokenizer
from collections import defaultdict
from dataclasses import dataclass
import itertools
import asyncio

from branch_reasoning.generation.branching import QueueDataset, get_new_branches
from branch_reasoning.generation.vllm_generation import (
    vLLM,
    vllm_generate_text,
    vllm_chat_generate_text,
    start_vllm,
    stop_vllm,
    kill_vllm_workers
)
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
    temperature: float = 1.0,
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
        
        completions = await vllm_chat_generate_text(
            vllm_instance,
            tokenizer,
            batch,
            num_completions=1,
            max_seq_len=max_len,
            generation_args=generation_args,
            temperature=temperature,
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
    model,
    tokenizer: PreTrainedTokenizer,
    dataset: itertools.cycle,
    total_completions: int,
    completions_per_prompt: int,
    gen_batch_size: int,
    current_iter: int,
    max_len: int,
    model_name: str = None,
    checkpoint_dir: str = "./checkpoints",
    vllm_server_args: Optional[Dict[str, Any]] = None,
    log_file: str = None,
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

    This function now handles:
    1. Saving the model and tokenizer to disk
    2. Starting and stopping the vLLM server
    3. Generating completions using vLLM
    """
    t0 = time.time()
    vllm_instance = None

    try:
        # First, save the model and tokenizer to disk
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Saving model and tokenizer to {checkpoint_dir}...")
        model_save_time = time.time()
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Model and tokenizer saved successfully in {time.time() - model_save_time:.2f}s")

        # Use the checkpoint directory as the model path for vLLM
        model_path = checkpoint_dir

        # Set up log file
        if log_file is None:
            log_file = f"./logs/vllm_gen_iter_{current_iter}.log"

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Set default vLLM server args if none are provided
        if vllm_server_args is None:
            vllm_server_args = {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.8,
                "max_model_len": max_len,
            }

        # Add a dynamic port to avoid conflicts - use iteration and current time to generate a port number
        # This ensures that each run gets a unique port
        vllm_server_args = vllm_server_args.copy()  # Create a copy to avoid modifying the original

        # Generate a port number based on current iteration and time - in range 10000-65000
        # Using both time and iteration ensures uniqueness even if multiple runs start at the same time
        port_base = 10000 + (current_iter % 1000)
        port_time_component = int(time.time() * 100) % 1000
        dynamic_port = port_base + port_time_component

        # Keep port in valid range (below 65535)
        if dynamic_port > 65000:
            dynamic_port = dynamic_port % 55000 + 10000

        vllm_server_args["port"] = dynamic_port

        print(f"Starting vLLM server with model from {model_path} on port {dynamic_port}...")

        # Kill any lingering processes first
        kill_vllm_workers()
        await asyncio.sleep(1)  # Wait a bit for resources to be released

        # Try up to 3 different ports if there's an issue
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                vllm_instance = await start_vllm(
                    model=model_path,
                    log_file=log_file,
                    max_concurrent_requests=gen_batch_size*2,
                    named_arguments=vllm_server_args,
                    verbosity=2,
                    timeout=300.0,  # 5 minutes timeout for server startup
                )
                print(f"vLLM server started successfully on port {vllm_server_args['port']}!")
                break  # Success, exit the loop
            except RuntimeError as e:
                if "is already in use" in str(e) and attempt < max_attempts - 1:
                    print(f"Port {vllm_server_args['port']} is already in use, trying another port...")
                    # Try a different port by adding 1000
                    vllm_server_args["port"] += 1000
                    if vllm_server_args["port"] > 65000:
                        vllm_server_args["port"] = vllm_server_args["port"] % 55000 + 10000
                    # Kill any processes on the previous port
                    kill_vllm_workers()
                    await asyncio.sleep(2)  # Give more time for resources to be released
                else:
                    raise  # Re-raise the exception if we've run out of attempts
        print("vLLM server started successfully!")

        # Calculate batching parameters
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
        all_bare_prompts = {}
        all_numbers = {}
        all_targets = {}
        all_tags = {}
        all_solutions = {}

        total_keys = 0

        # Generate completions
        for i in range(generation_iter):
            prompts, numbers, targets, tags, solutions, bare_prompts = _fetch_batch(
                dataset, prompts_per_batch
            )

            for j in range(prompt_repetitions):
                # Use vLLM to generate completions
                completions = await vllm_chat_generate_text(
                    vllm_instance,
                    tokenizer,
                    prompts,
                    num_completions=num_completions,
                    max_seq_len=max_len,
                    generation_args=generation_args,
                    temperature=temperature,
                )

                if wandb_logging and j == 0:
                    # Log example completion
                    example_completion = prompts[0] + completions[0]
                    example_completion_html = wandb.Html(
                        f"<pre>{html.escape(example_completion)}</pre>"
                    )
                    example_completion_text = wandb.Html(f"```\n{example_completion}\n```")

                for k in range(prompts_per_batch):
                    total_keys += 1
                    base_key = f"{total_keys}#{current_iter}"
                    all_prompts[base_key] = prompts[k]
                    all_bare_prompts[base_key] = bare_prompts[k]
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
                temperature,
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
        result = _pack_into_return_datatypes(
            all_completions,
            all_prompts,
            all_numbers,
            all_targets,
            all_tags,
            all_solutions,
            all_bare_prompts,
        )

        print(f"Generation Time taken: {time.time() - t0} seconds")
        return result

    except Exception as e:
        print(f"Error in vllm_generate_completions: {e}")
        raise

    finally:
        # Stop the vLLM server
        if vllm_instance is not None:
            print("Stopping vLLM server...")
            try:
                # Get the port that was used
                used_port = vllm_server_args.get("port", None)

                # Normal stop procedure
                await stop_vllm(vllm_instance)
                print("vLLM server stopped successfully!")

                # Additionally terminate processes by port if a port was specified
                if used_port:
                    try:
                        import subprocess
                        # Find and kill processes using the port
                        cmd = f"lsof -ti:{used_port} | xargs kill -9"
                        subprocess.run(
                            cmd,
                            shell=True,
                            check=False,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL
                        )
                        print(f"Killed any processes using port {used_port}")
                    except Exception as port_error:
                        print(f"Error killing processes on port {used_port}: {port_error}")
            except Exception as e:
                print(f"Error stopping vLLM server: {e}")
            finally:
                # Always try to kill any remaining workers
                kill_vllm_workers()
                await asyncio.sleep(2)  # Allow time for resources to be released


if __name__ == "__main__":
    pass