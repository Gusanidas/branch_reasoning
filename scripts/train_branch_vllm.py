#!/usr/bin/env python
import time
import asyncio
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from collections import defaultdict
from datasets import concatenate_datasets, load_dataset
from itertools import cycle
import os
import sys
import signal
import traceback
import argparse
from typing import Optional, Dict, Any, List
import subprocess
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from branch_reasoning.scoring import CompletionScorer, match_format_exactly, rate_countdown_answer, match_format_approximately
from branch_reasoning.utils.countdown_task import transform_countdown_data, apply_r1_template
from branch_reasoning.utils.prompts import (
    base_prompt,
    single_branch_examples,
    multi_branch_examples,
    single_branch_format_prompt,
    multi_branch_format_prompt,
)
from branch_reasoning.utils.utils import linear_warmup_decay
from branch_reasoning.models.model_loader import get_models_and_tokenizers
from branch_reasoning.generation.completions import generate_completions
from branch_reasoning.log_probs import populate_log_probs
from branch_reasoning.training.trainer import train_with_grpo

from branch_reasoning.generation.vllm_generation import (
    start_vllm, 
    stop_vllm, 
    kill_vllm_workers,
    vLLM
)
from branch_reasoning.generation.vllm_completions import vllm_generate_completions

# Global variable to track the vLLM instance for cleanup during signals
current_vllm_instance: Optional[vLLM] = None

def signal_handler(sig, frame):
    """Signal handler for graceful shutdown."""
    print(f"\n\nReceived signal {sig}, shutting down...")
    
    if current_vllm_instance is not None:
        # We need to run the async cleanup in a new event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(cleanup_vllm(current_vllm_instance))
        finally:
            loop.close()
    
    kill_vllm_workers()
    
    if wandb.run is not None:
        wandb.finish()
        
    print("All processes terminated, exiting...")
    sys.exit(0)

async def cleanup_vllm(vllm_instance: vLLM):
    """Cleanup function to ensure vLLM resources are properly released."""
    try:
        await stop_vllm(vllm_instance)
        print("vLLM server stopped successfully during cleanup")
    except Exception as e:
        print(f"Error during vLLM cleanup: {e}")
        # Ensure workers are killed even if stop_vllm fails
        kill_vllm_workers()

async def train_branch(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    reference_model_name: Optional[str] = None,
    use_vllm: bool = False,
    device: str = "cuda",
    save_interval: int = 20,
    hf_repo_name: str = "Gusanidas/branch-grpo-model-qwen-3b-branch",
    hf_dataset_name: str = "Gusanidas/countdown-tasks-dataset-balanced",
    iterations: int = 500,
    total_completions: int = 2,
    gen_batch_size: int = 2,
    no_completions: int = 2,
    branching_factor: int = 2,
    max_branching_points: int = 3,
    max_len: int = 768,
    opt_generation_args: Dict[str, Any] = None,
    wandb_logging: bool = False,
    branch_completions: bool = True,
    use_bfloat16: bool = True,
    vllm_server_args: Dict[str, Any] = None,
    wandb_project_name: str = "branch_grpo_vast_branch",
    run_name: str = "vllm_run_1",
    training_args: Dict[str, Any] = None,
    temperature: float = 1.0,
):
    """
    Unified training function that can use either vLLM or HuggingFace for text generation.
    
    Args:
        model_name: Name of the model to use
        tokenizer_name: Name of the tokenizer to use (if None, uses model_name)
        reference_model_name: Name of the reference model for RLHF (if None, uses model_name)
        use_vllm: Whether to use vLLM for text generation
        device: Device to use for model (cuda, cpu, mps)
        save_interval: How often to save model checkpoints
        hf_repo_name: HuggingFace repository name for saving
        hf_dataset_name: HuggingFace dataset name to load
        iterations: Number of training iterations
        total_completions: Total number of completions to generate
        gen_batch_size: Batch size for generation
        no_completions: Number of completions per prompt
        branching_factor: Number of branches per branch point
        max_branching_points: Maximum number of branch points
        max_len: Maximum sequence length
        opt_generation_args: Arguments for text generation
        wandb_logging: Whether to log to Weights & Biases
        branch_completions: Whether to generate branched completions
        use_bfloat16: Whether to use bfloat16 precision
        vllm_server_args: Additional arguments for vLLM server
    """
    global current_vllm_instance
    
    # Set default parameters if not provided
    if opt_generation_args is None:
        opt_generation_args = {
            "top_p": 1,
            "top_k": 50,
        }
    
    if use_vllm and vllm_server_args is None:
        vllm_server_args = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 2048,
            "disable_log_requests": True,
            "enable_prefix_caching": True,
            "enforce_eager": True,
        }
    
    vllm_instance = None
    wandb_run_started = False
    
    try:
        # Validate parameters
        if no_completions % gen_batch_size != 0 and gen_batch_size % no_completions != 0:
            raise ValueError(
                "no_completions must be divisible by gen_batch_size or gen_batch_size must be divisible by no_completions"
            )

        if no_completions > gen_batch_size:
            prompt_repetitions = no_completions // gen_batch_size
            prompts_per_batch = 1
        else:
            prompt_repetitions = 1
            prompts_per_batch = gen_batch_size // no_completions #TODO: Do inside func

        print(f"prompts_per_batch: {prompts_per_batch}")
        print(f"prompt_repetitions: {prompt_repetitions}")

        if total_completions % (prompts_per_batch * prompt_repetitions) != 0:
            raise ValueError(
                "total_completions must be divisible by prompts_per_batch * prompt_repetitions"
            )

        generation_iter = total_completions // (gen_batch_size * prompt_repetitions)
        print(f"generation_iter: {generation_iter}")

        # Ensure log directory exists
        os.makedirs("./logs", exist_ok=True)

        # Load the model and tokenizer using get_models_and_tokenizers for both vLLM and HuggingFace
        print(f"Loading model and tokenizers from {model_name}...")
        beta = training_args.get("beta", 0.01) if training_args is not None else 0.01
        print(f"beta: {beta}, training_args: {training_args}")
        model, reference_model, tokenizer = get_models_and_tokenizers(
            model_name, 
            reference_model_name, 
            tokenizer_name, 
            use_bfloat16, 
            beta=beta,  # Use beta from training_args
            use_vllm=use_vllm  # Pass the use_vllm flag
        )
        print(f"reference_model: {reference_model}")
        
        # Create optimizer and scheduler outside the loop
        learning_rate = training_args.get("learning_rate", 5e-6) if training_args is not None else 5e-6
        weight_decay = training_args.get("weight_decay", 0.01) if training_args is not None else 0.01
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Create learning rate scheduler
        # Calculate total steps for the scheduler
        total_steps = iterations * training_args.get("num_epochs", 1) #TODO: change this
        warmup_steps = max(1, int(total_steps * 0.1))  # 10% warmup
        scheduler = linear_warmup_decay(optimizer, warmup_steps, total_steps)

        # Load dataset
        print(f"Loading dataset from {hf_dataset_name}...")
        task_dataset = load_dataset(hf_dataset_name)["train"]
        examples = ["Example:\n" + example for example in single_branch_examples] + [""]*5
        dataset = transform_countdown_data(
            task_dataset, 
            base_prompt_template=base_prompt, 
            template_func=apply_r1_template, 
            #format_prompt=multi_branch_format_prompt, 
            format_prompt=single_branch_format_prompt,
            examples=examples
        )
        cycling_dataset = cycle(dataset)
        print(f"Dataset loaded and transformed successfully.")

        completion_scorer = CompletionScorer(
            scoring_functions=[
                match_format_exactly,
                rate_countdown_answer,
                match_format_approximately,
            ]
        )

        # Initialize wandb if needed
        if wandb_logging:
            try:
                print("Initializing wandb...")
                wandb.init(
                    project=wandb_project_name,
                    name=run_name,
                    config={
                        "model_name": model_name,
                        "tokenizer_name": tokenizer_name,
                        "use_vllm": use_vllm,
                        "iterations": iterations,
                        "total_completions": total_completions,
                        "gen_batch_size": gen_batch_size,
                        "no_completions": no_completions,
                        "prompts_per_batch": prompts_per_batch,
                        "prompt_repetitions": prompt_repetitions,
                        "generation_iter": generation_iter,
                        "vllm_server_args": vllm_server_args if use_vllm else None,
                        "training_args": training_args,
                        "learning_rate": training_args.get("learning_rate", 5e-5) if training_args is not None else 5e-5,
                    },
                )
                wandb_run_started = True
                print("wandb initialized successfully.")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                wandb_logging = False  # Disable wandb logging if it fails

        t0 = time.time()
        t2, t3 = t0, t0

        for i in range(iterations):
            time.sleep(20)
            try:
                wandb_logs = defaultdict(float)
                print(f"=== Starting iteration {i+1}/{iterations} ===, time: {time.time() - t0:.2f}s")
                iter_start_time = time.time()
                
                # If using vLLM, start the server with the current model weights for this iteration
                if use_vllm:
                    # Make sure to kill any previous vLLM instance and workers
                    if vllm_instance is not None:
                        print(f"Stopping vLLM server from previous iteration...")
                        try:
                            await stop_vllm(vllm_instance)
                        except Exception as e:
                            print(f"Error stopping vLLM server: {e}")
                        finally:
                            # Always forcefully kill workers regardless of stop_vllm success
                            kill_vllm_workers()
                            # Allow time for port to be released
                            await asyncio.sleep(2)
                            vllm_instance = None
                            current_vllm_instance = None
                    else:
                        # Even if we don't have a vllm_instance, there might be lingering processes
                        kill_vllm_workers()
                        # Allow time for port to be released
                        await asyncio.sleep(2)
                    
                    # Use checkpoints directory if it exists for iterations after the first
                    # For the first iteration, we use the original model
                    checkpoint_dir = "./checkpoints"
                    model_to_use = model_name
                    if i > 0 and os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
                        model_to_use = checkpoint_dir
                        print(f"Starting vLLM server with model from checkpoint directory: {checkpoint_dir}")
                    else:
                        print(f"Starting vLLM server with model {model_name}...")
                    
                    try:
                        # We need to change the port for each iteration to avoid conflicts
                        iter_vllm_server_args = vllm_server_args.copy()
                        iter_vllm_server_args["port"] = 8000 + (i % 10)  # Use different ports across iterations
                        
                        vllm_instance = await start_vllm(
                            model=model_to_use,
                            log_file=f"./logs/vllm_train_iter_{i}.log",
                            max_concurrent_requests=gen_batch_size*2,
                            named_arguments=iter_vllm_server_args,
                            verbosity=2,
                            timeout=300.0,  # 5 minutes timeout for server startup
                        )
                        # Set the global instance for signal handling
                        current_vllm_instance = vllm_instance
                        print("vLLM server started successfully!")
                    except Exception as e:
                        print(f"Error starting vLLM server: {e}")
                        traceback.print_exc()
                        # Make sure to clean up any processes that might have been started
                        kill_vllm_workers()
                        # Allow time for port to be released
                        await asyncio.sleep(2)
                        continue  # Skip this iteration and try the next one
                
                # Generate completions using either vLLM or HuggingFace
                try:
                    print(f"Generating completions for iteration {i+1}...")
                    if use_vllm:
                        # Use vLLM for generation
                        prompt_completions = await vllm_generate_completions(
                            vllm_instance=vllm_instance,
                            tokenizer=tokenizer,
                            dataset=cycling_dataset,
                            total_completions=total_completions,
                            completions_per_prompt=no_completions,
                            gen_batch_size=gen_batch_size,
                            current_iter=i,
                            max_len=max_len,
                            generation_args=opt_generation_args,
                            wandb_logging=wandb_logging,
                            branch_completions=branch_completions,
                            branching_factor=branching_factor,
                            max_branching_points=max_branching_points,
                            temperature=temperature,
                        )
                    else:
                        # Use HuggingFace for generation
                        prompt_completions = generate_completions(
                            model,
                            tokenizer,
                            cycling_dataset,
                            total_completions,
                            completions_per_prompt=no_completions,
                            gen_batch_size=gen_batch_size,
                            current_iter=i,
                            max_len=max_len,
                            generation_args=opt_generation_args,
                            device=device,
                            wandb_logging=wandb_logging,
                            branch_completions=branch_completions,
                            branching_factor=branching_factor,
                            max_branching_points=max_branching_points,
                            temperature=temperature,
                        )
                    
                    t2 = time.time()
                    iter_time = time.time() - iter_start_time
                    print(f"Iteration {i+1} completed in {iter_time:.2f}s")
                    wandb_logs["iter_time"] = iter_time

                    prompt_completions = completion_scorer.score_completions(
                        prompt_completion_list=prompt_completions,
                        wandb_logs=wandb_logs,
                    )
                    
                    populate_log_probs(
                        prompt_completions=prompt_completions,
                        model=model,
                        tokenizer=tokenizer,
                        reference_model=reference_model,
                        device=device,
                    )
                    t3 = time.time()
                    wandb_logs["populate_log_probs_time"] = t3 - t2

                    print("Starting GRPO training...")
                    train_stats = train_with_grpo(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_completions=prompt_completions,
                        optimizer=optimizer,
                        scheduler=scheduler, 
                        device=device,
                        iteration=i,
                        use_wandb=wandb_logging,
                        training_args=training_args,
                    )
                    wandb_logs["train_time"] = time.time() - t3
                    print(f"GRPO training completed for iteration {i+1}")
                    print(f"Final policy loss: {train_stats['policy_losses'][-1] if train_stats['policy_losses'] else 'N/A'}")
                    
                    # Process and print results
                    print(f"Len of prompt_completions: {len(prompt_completions)}")
                    #for prompt_completion in prompt_completions:
                    #    print("-" * 100)
                    #    print(f"PROMPT")
                    #    print(prompt_completion.prompt)
                    #    print(prompt_completion.metadata.solution)
                    #    print(f"len of branched_completions: {len(prompt_completion.branched_completions)}")
                    #    branched_completions = prompt_completion.branched_completions
                    #    for branched_completion in branched_completions:
                    #        print("-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                    #        print(f"BRANCHED COMPLETION")
                    #        print(f"len of branches: {len(branched_completion.branches)}")
                    #        branches = branched_completion.branches
                    #        for branch in branches:
                    #            print("-" * 100)
                    #            print(f"BRANCH")
                    #            print(branch.completion)
                    #            print(f"score: {branch.score}")
                    #            print(f"log_probs: {branch.log_probs[:10]}")
                    #            print(f"ref_log_probs: {branch.ref_log_probs[:10]}")
                    #            print("-" * 100)
                    #        print("-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                    #    print("^^" * 100)

                    if wandb_logging:
                        for key, value in wandb_logs.items():
                            if key + "_steps" in wandb_logs:
                                v = value/wandb_logs[key+"_steps"]
                                wandb.log({key: v}, step=i)
                            elif not key.endswith("_steps"):
                                wandb.log({key: value}, step=i)
                
                except Exception as e:
                    print(f"Error in iteration {i+1}: {e}")
                    traceback.print_exc()
                    # Continue with next iteration if there's an error
                    print(f"Skipping to next iteration...")
                    continue
                
                # If using vLLM, save the model weights and stop the server after the iteration
                if use_vllm:
                    print(f"Stopping vLLM server after iteration {i+1}...")
                    if vllm_instance is not None:
                        try:
                            # First stop the vLLM server
                            await stop_vllm(vllm_instance)
                            print("vLLM server stopped successfully!")
                            
                            # After stopping vLLM, save the model weights for the next iteration
                            print(f"Saving model weights to ./checkpoints for iteration {i+1}...")
                            
                            # Load the model temporarily to save it to the checkpoints directory
                            # Note: We only need to do this if we're using vLLM, as otherwise
                            # the model is already loaded in memory
                            os.makedirs("./checkpoints", exist_ok=True)
                            
                            # For the first iteration, we need to load the model first
                            # Later iterations would have used the checkpoint already
                            temp_model = None
                            try:
                                if i == 0:  # First iteration
                                    if use_bfloat16:
                                        temp_model = AutoModelForCausalLM.from_pretrained(
                                            model_name, torch_dtype=torch.bfloat16
                                        )
                                    else:
                                        temp_model = AutoModelForCausalLM.from_pretrained(model_name)
                                else:  # Subsequent iterations
                                    if use_bfloat16:
                                        temp_model = AutoModelForCausalLM.from_pretrained(
                                            "./checkpoints", torch_dtype=torch.bfloat16
                                        )
                                    else:
                                        temp_model = AutoModelForCausalLM.from_pretrained("./checkpoints")
                                
                                # Save the model to the checkpoints directory
                                temp_model.save_pretrained("./checkpoints")

                                # Also save the tokenizer to the same directory
                                tokenizer.save_pretrained("./checkpoints")
                                print(f"Model weights and tokenizer saved successfully to ./checkpoints")

                                # Free up memory
                                del temp_model
                                torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"Error saving model weights: {e}")
                                
                        except Exception as e:
                            print(f"Error stopping vLLM server: {e}")
                        finally:
                            # Always forcefully kill workers regardless of stop_vllm success
                            kill_vllm_workers()
                            # Allow time for resources to be released
                            await asyncio.sleep(2)
                            vllm_instance = None
                            current_vllm_instance = None
                    else:
                        # Even if we don't have a vllm_instance, there might be lingering processes
                        kill_vllm_workers()
                        await asyncio.sleep(2)
            
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                break
            except Exception as e:
                print(f"Unexpected error in iteration {i+1}: {e}")
                traceback.print_exc()
                break
    
    except Exception as e:
        print(f"Unexpected error in main function: {e}")
        traceback.print_exc()
            
    finally:
        # Clean up resources
        if use_vllm:
            print("\nFinal cleanup: Stopping vLLM server...")
            if vllm_instance is not None:
                try:
                    await stop_vllm(vllm_instance)
                    print("vLLM server stopped successfully!")
                except Exception as e:
                    print(f"Error stopping vLLM server: {e}")
                    traceback.print_exc()
                finally:
                    # Force kill any remaining vLLM workers regardless of stop_vllm success
                    kill_vllm_workers()
                    await asyncio.sleep(2)  # Allow time for resources to be released
            
            # Reset variables
            vllm_instance = None
            current_vllm_instance = None
            
            # Double-check that no vLLM processes are left running with multiple attempts
            print("Final cleanup: Ensuring all vLLM processes are terminated...")
            for _ in range(3):  # Try multiple times to ensure cleanup
                kill_vllm_workers()
                await asyncio.sleep(1)  # Give some time between kill attempts
        
        # Finish wandb if it was started
        if wandb_run_started and wandb.run is not None:
            try:
                wandb.finish()
                print("wandb run finished.")
            except Exception as e:
                print(f"Error finishing wandb run: {e}")
        
        print("All cleanup complete")


async def main():
    """Main function to run training."""
    # Fixed parameters
    save_interval = 20
    hf_repo_name = "Gusanidas/branch-grpo-model-qwen-3b-branch"
    hf_dataset_name = "Gusanidas/countdown-tasks-dataset-easier"

    iterations = 125

    # Generation parameters
    # In one iteration, how many solutions to generate. Several branches can belong to the same completion.
    total_completions = 64
    # How many completions to generate in one batch.
    gen_batch_size = 64
    # How many completions to generate per prompt.
    no_completions = 8
    # How many branches to generate for each branching point.
    branching_factor = 2
    # Maximum number of branching points.
    max_branching_points = 3
    # Maximum length of the completion.
    max_len = 1024
    # Generation arguments.
    temperature = 1.0
    opt_generation_args = {
        "top_p": 1,
        "top_k": 50,
    }

    use_vllm = True
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    tokenizer_name = None
    device = "cuda"
    wandb_logging = True
    branch_completions = False
    use_bfloat16 = True

    wandb_project_name = "branch_grpo_vast_branch_vllm"
    run_name = "run_23"
    
    training_args = {
        "num_epochs": 2,
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "max_grad_norm": 3.0,
        "beta": 0.01,
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 8,
    }

    vllm_server_args = {
        "host": "0.0.0.0",
        "port": 8000,
        "served_model_name": model_name,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": max_len,
    }
    
    # Set device based on availability
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Handle tokenizer name
    tokenizer_name = tokenizer_name or model_name
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Start timing the full execution
    total_start_time = time.time()
    
    # Run the training function
    await train_branch(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=device,
        use_vllm=use_vllm,
        save_interval=save_interval,
        hf_repo_name=hf_repo_name,
        hf_dataset_name=hf_dataset_name,
        iterations=iterations,
        total_completions=total_completions,
        gen_batch_size=gen_batch_size,
        no_completions=no_completions,
        branching_factor=branching_factor,
        max_branching_points=max_branching_points,
        max_len=max_len,
        opt_generation_args=opt_generation_args,
        wandb_logging=wandb_logging,
        branch_completions=branch_completions,
        use_bfloat16=use_bfloat16,
        wandb_project_name=wandb_project_name,
        run_name=run_name,
        training_args=training_args,
        temperature=temperature,
        #vllm_server_args=vllm_server_args,
    )
    
    # Calculate and print the total execution time
    total_execution_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
    print(f"{'='*50}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error running main function: {e}")
        traceback.print_exc()
    finally:
        # Calculate and print execution time even if there was an error
        if 'total_start_time' in globals():
            total_execution_time = time.time() - total_start_time
            print(f"\n{'='*50}")
            print(f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
            print(f"{'='*50}")
            
        print("Emergency cleanup: Ensuring all vLLM processes are terminated...")
        # Try multiple times to ensure cleanup
        for i in range(5):
            kill_vllm_workers()
            time.sleep(1)  # Give some time between kill attempts
            # Try killing by port as well
            for port in range(8000, 8010):
                try:
                    # Attempt to find and kill processes using these ports
                    subprocess.run(
                        f"lsof -ti:{port} | xargs kill -9",
                        shell=True, 
                        check=False, 
                        stderr=subprocess.DEVNULL, 
                        stdout=subprocess.DEVNULL
                    )
                except:
                    pass
            time.sleep(0.5)
        
        # Finish wandb if it was started and not already finished
        if wandb.run is not None:
            try:
                wandb.finish()
            except:
                pass
                
        print("Final cleanup complete")