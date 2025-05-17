#!/usr/bin/env python
import time
import gc
import asyncio
import random
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
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

from branch_reasoning.utils.utils import (
    _print_gpu_memory,
    move_optimizer_state,
    linear_warmup_decay,
)

from branch_reasoning.scoring import (
    CompletionScorer,
    match_format_exactly,
    rate_countdown_answer,
    match_format_approximately,
    match_format_loosely,
    score_branch_format_approx,
    score_branch_format_loose,
    score_branch_format,
)
from branch_reasoning.countdown_task import transform_countdown_data, apply_r1_template
from branch_reasoning.prompts import (
    base_prompt,
    get_format_and_examples,
)
from branch_reasoning.models.model_loader import get_models_and_tokenizers
from branch_reasoning.generation.completions import generate_completions
from branch_reasoning.log_probs import populate_log_probs
from branch_reasoning.training.trainer import train_with_grpo

from branch_reasoning.generation.vllm_generation import (
    start_vllm,
    stop_vllm,
    kill_vllm_workers,
    vLLM,
)
from branch_reasoning.generation.vllm_completions import vllm_generate_completions
from branch_reasoning.reuse_completions import ReuseCompletionsDataset

# No global instance needed anymore


def signal_handler(sig, frame):
    """Signal handler for graceful shutdown."""
    print(f"\n\nReceived signal {sig}, shutting down...")

    # We need to run the async cleanup in a new event loop
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(cleanup_after_training())
    finally:
        loop.close()

    if wandb.run is not None:
        wandb.finish()

    print("All processes terminated, exiting...")
    sys.exit(0)


async def cleanup_after_training():
    """Cleanup function to ensure all resources are properly released."""
    # Ensure workers are killed
    kill_vllm_workers()

    # Wait a moment for processes to terminate
    await asyncio.sleep(1)

    # Try killing processes using ports in range 10000-65000 (our dynamic port range)
    try:
        # Use a more targeted approach to kill processes on the ports we might be using
        for port_range_start in range(10000, 63000, 2000):
            port_range_end = min(port_range_start + 2000, 65000)
            command = f"for port in $(seq {port_range_start} {port_range_end}); do lsof -ti:$port | xargs kill -9 2>/dev/null || true; done"
            subprocess.run(
                command,
                shell=True,
                check=False,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        # Also target specific vLLM processes
        subprocess.run(
            "pkill -f 'vllm.entrypoints.openai.api_server' || true",
            shell=True,
            check=False,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

        # Kill any hanging Python multiprocessing workers
        subprocess.run(
            "pkill -f 'multiprocessing.spawn import spawn_main' || true",
            shell=True,
            check=False,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Error during additional cleanup: {e}")


def _get_optimizer_and_scheduler(
    model: PreTrainedModel,
    training_args: Dict[str, Any],
    iterations: int,
    no_completions: int,
):
    # Create optimizer and scheduler outside the loop
    learning_rate = (
        training_args.get("learning_rate", 5e-6) if training_args is not None else 5e-6
    )
    weight_decay = (
        training_args.get("weight_decay", 0.01) if training_args is not None else 0.01
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_batch_size = training_args.get("batch_size", 1)
    total_steps_default = (
        iterations
        * training_args.get("num_epochs", 1)
        * (no_completions / train_batch_size)
    )
    total_steps = training_args.get("total_steps", total_steps_default)

    warmup_steps_default = max(1, int(total_steps * 0.1))  # 10% warmup
    warmup_steps = training_args.get("warmup_steps", warmup_steps_default)
    scheduler = linear_warmup_decay(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def _double_gradient_accumulation(
    training_args: Dict[str, Any], no_completions: int, total_completions: int
):
    max_grad_acc = training_args.get(
        "max_gradient_accumulation_steps", total_completions // no_completions
    )
    training_args["gradient_accumulation_steps"] = min(
        2 * training_args.get("gradient_accumulation_steps", 1), max_grad_acc
    )


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
    reuse_completions: bool = False,
    reuse_completions_batch_size: int = 1,
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
        wandb_project_name: Name of the wandb project
        run_name: Name of the wandb run
        training_args: Training arguments
        temperature: Temperature for generation
        reuse_completions: Whether to reuse completions
        reuse_completions_batch_size: Batch size for reusing completions
    """
    wandb_run_started = False

    try:
        os.makedirs("./logs", exist_ok=True)

        print(f"Loading model and tokenizers from {model_name}...")
        beta = training_args.get("beta", 0.01) if training_args is not None else 0.01
        print(f"beta: {beta}, training_args: {training_args}")
        model, reference_model, tokenizer = get_models_and_tokenizers(
            model_name,
            reference_model_name,
            tokenizer_name,
            use_bfloat16,
            beta=beta,
            use_vllm=use_vllm,
        )

        optimizer, scheduler = _get_optimizer_and_scheduler(
            model, training_args, iterations, no_completions
        )
        # Load dataset
        print(f"Loading dataset from {hf_dataset_name}...")
        task_dataset = load_dataset(hf_dataset_name)["train"]
        format_prompt, examples = get_format_and_examples(
            branch_completions, max_branching_points, branching_factor
        )
        examples = ["Example:\n" + example for example in examples]
        examples = examples + [""] * len(examples)
        dataset = transform_countdown_data(
            task_dataset,
            base_prompt_template=base_prompt,
            template_func=None,
            format_prompt=format_prompt,
            examples=examples,
        )
        cycling_dataset = cycle(dataset)
        print(f"Dataset loaded and transformed successfully.")
        pscore_branch_format_approx = partial(
            score_branch_format_approx,
            max_branches=max_branching_points,
            branch_factor=branching_factor,
        )
        pscore_branch_format_loose = partial(
            score_branch_format_loose,
            max_branches=max_branching_points,
            branch_factor=branching_factor,
        )
        pscore_branch_format = partial(
            score_branch_format,
            max_branches=max_branching_points,
            branch_factor=branching_factor,
        )  # TODO: consistent naming (branch_factor, max_branche,...)

        pscore_branch_format_approx.__name__ = "score_branch_format_approx"
        pscore_branch_format_loose.__name__ = "score_branch_format_loose"
        pscore_branch_format.__name__ = "score_branch_format"

        completion_scorer = CompletionScorer(
            scoring_functions=[
                match_format_exactly,
                rate_countdown_answer,
                match_format_approximately,
                match_format_loosely,
                pscore_branch_format_approx,
                pscore_branch_format_loose,
                pscore_branch_format,
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
                        "vllm_server_args": vllm_server_args if use_vllm else None,
                        "training_args": training_args,
                        "learning_rate": training_args.get("learning_rate", 5e-5)
                        if training_args is not None
                        else 5e-5,
                    },
                )
                wandb_run_started = True
                print("wandb initialized successfully.")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                wandb_logging = False

        t0 = time.time()
        t2, t3 = t0, t0

        if reuse_completions:
            reuse_completions_dataset = ReuseCompletionsDataset(
                max_length=total_completions * 5
            )
        for i in range(iterations):
            wandb_logs = defaultdict(float)
            print(
                f"=== Starting iteration {i+1}/{iterations} ===, time: {time.time() - t0:.2f}s"
            )
            iter_start_time = time.time()

            try:
                print(f"Generating completions for iteration {i+1}...")
                print(f"Device of model: {model.device}")
                if use_vllm:
                    # Add retry mechanism for vLLM generation
                    max_retries = 3
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            prompt_completions = await vllm_generate_completions(
                                model=model,
                                tokenizer=tokenizer,
                                dataset=cycling_dataset,
                                total_completions=total_completions,
                                completions_per_prompt=no_completions,
                                gen_batch_size=gen_batch_size,
                                current_iter=i,
                                max_len=max_len,
                                model_name=model_name,
                                checkpoint_dir="./checkpoints",
                                vllm_server_args=vllm_server_args,
                                log_file=f"./logs/vllm_train_iter_{i}.log",
                                generation_args=opt_generation_args,
                                wandb_logging=wandb_logging,
                                branch_completions=branch_completions,
                                branching_factor=branching_factor,
                                max_branching_points=max_branching_points,
                                temperature=temperature,
                            )
                            break  # Success, exit retry loop
                        except Exception as vllm_error:
                            retry_count += 1
                            print(
                                f"vLLM generation error (attempt {retry_count}/{max_retries}): {vllm_error}"
                            )

                            # Clean up vLLM processes before retry
                            try:
                                kill_vllm_workers()
                                await asyncio.sleep(
                                    2
                                )  # Give more time for processes to terminate
                            except Exception as cleanup_error:
                                print(f"Error during vLLM cleanup: {cleanup_error}")

                            if retry_count >= max_retries:
                                print(
                                    "Maximum retries reached for vLLM generation, raising error"
                                )
                                raise
                            else:
                                print(f"Retrying vLLM generation in 5 seconds...")
                                await asyncio.sleep(5)  # Wait before retry
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

                if reuse_completions:
                    reuse_completions_dataset.add_completions(prompt_completions)

                if i > 9 and reuse_completions:
                    reuse_completions_batch = reuse_completions_dataset.next_batch(
                        reuse_completions_batch_size
                    )
                    prompt_completions = prompt_completions + reuse_completions_batch
                    random.shuffle(prompt_completions)

                t2 = time.time()
                iter_time = time.time() - iter_start_time
                print(f"Iteration {i+1} completed in {iter_time:.2f}s")
                wandb_logs["iter_time"] = iter_time

                prompt_completions = completion_scorer.score_completions(
                    prompt_completion_list=prompt_completions,
                    wandb_logs=wandb_logs,
                )

                # Add retry mechanism for log probs calculation
                max_log_probs_retries = 2
                log_probs_retry_count = 0
                while log_probs_retry_count < max_log_probs_retries:
                    try:
                        populate_log_probs(
                            prompt_completions=prompt_completions,
                            model=model,
                            tokenizer=tokenizer,
                            batch_size=training_args.get("log_probs_batch_size", 1),
                            reference_model=reference_model,
                            device=device,
                        )
                        t3 = time.time()
                        wandb_logs["populate_log_probs_time"] = t3 - t2
                        break  # Success, exit retry loop
                    except Exception as log_probs_error:
                        log_probs_retry_count += 1
                        print(
                            f"Error calculating log probabilities (attempt {log_probs_retry_count}/{max_log_probs_retries}): {log_probs_error}"
                        )

                        # Try to clean up memory
                        try:
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception as cleanup_error:
                            print(f"Error during memory cleanup: {cleanup_error}")

                        if log_probs_retry_count >= max_log_probs_retries:
                            print(
                                "Maximum retries reached for log probs calculation, raising error"
                            )
                            raise
                        else:
                            print(
                                f"Retrying log probs calculation with smaller batch size..."
                            )
                            # Reduce batch size for retry
                            reduced_batch_size = max(
                                1, training_args.get("log_probs_batch_size", 1) // 2
                            )
                            training_args["log_probs_batch_size"] = reduced_batch_size
                            print(f"Reduced batch size to {reduced_batch_size}")
                            await asyncio.sleep(2)  # Wait before retry

                move_optimizer_state(optimizer, device)
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
                model = model.to("cpu")
                if reference_model is not None:
                    reference_model = reference_model.to("cpu")
                torch.cuda.empty_cache()
                del prompt_completions
                gc.collect()
                torch.cuda.empty_cache()
                wandb_logs["train_time"] = time.time() - t3
                print(f"GRPO training completed for iteration {i+1}")
                print(
                    f"Final policy loss: {train_stats['policy_losses'][-1] if train_stats['policy_losses'] else 'N/A'}"
                )
                if i % 2 == 1:
                    _double_gradient_accumulation(
                        training_args, no_completions, total_completions
                    )

                if wandb_logging:
                    for key, value in wandb_logs.items():
                        if key + "_steps" in wandb_logs:
                            v = value / wandb_logs[key + "_steps"]
                            wandb.log({key: v}, step=i)
                        elif not key.endswith("_steps"):
                            wandb.log({key: value}, step=i)

            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                break
            except Exception as e:
                print(f"Unexpected error in iteration {i+1}: {e}")
                traceback.print_exc()

                # Clean up resources before attempting next iteration
                print(f"Cleaning up resources after error in iteration {i+1}...")

                # Move models to CPU to free GPU memory
                try:
                    model = model.to("cpu")
                    if reference_model is not None:
                        reference_model = reference_model.to("cpu")
                except Exception as cleanup_e:
                    print(f"Error moving models to CPU: {cleanup_e}")

                # Clear CUDA cache
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as cleanup_e:
                    print(f"Error clearing CUDA cache: {cleanup_e}")

                # Kill vLLM workers if using vLLM
                if use_vllm:
                    try:
                        kill_vllm_workers()
                        time.sleep(1)  # Give some time for processes to terminate
                    except Exception as cleanup_e:
                        print(f"Error killing vLLM workers: {cleanup_e}")

                # Log the error to wandb if enabled
                if wandb_logging and wandb.run is not None:
                    try:
                        wandb.log({"iteration_error": str(e)}, step=i)
                    except Exception as cleanup_e:
                        print(f"Error logging to wandb: {cleanup_e}")

                print(f"Cleanup complete. Continuing to next iteration...")
                continue  # Continue to next iteration instead of breaking

    except Exception as e:
        print(f"Unexpected error in main function: {e}")
        traceback.print_exc()

    finally:
        if use_vllm:
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
    hf_dataset_name = "Gusanidas/countdown-tasks-dataset-balanced"

    iterations = 300

    # Generation parameters
    # In one iteration, how many solutions to generate. Several branches can belong to the same completion.
    total_completions = 64
    # How many completions to generate in one batch.
    gen_batch_size = 32
    # How many completions to generate per prompt.
    no_completions = 8
    # How many branches to generate for each branching point.
    branching_factor = 2
    # Maximum number of branching points.
    max_branching_points = 2
    # Maximum length of the completion.
    max_len = 1536
    # Generation arguments.
    temperature = 1.0
    opt_generation_args = {
        "top_p": 0.9,
        # "top_k": 50,
        "max_tokens": max_len,
    }

    use_vllm = True
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    #model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    # model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-1.7B"
    tokenizer_name = None
    device = "cuda"
    wandb_logging = True
    branch_completions = True
    use_bfloat16 = True

    reuse_completions = True
    reuse_completions_batch_size = 4

    wandb_project_name = "branch_grpo_vast_branch_vllm"
    run_name = "run_162"

    training_args = {
        "num_epochs": 2,
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "max_grad_norm": 2.0,
        "beta": 0.01,
        "learning_rate": 9e-6,
        "weight_decay": 0.05,
        "gradient_accumulation_steps": 2,
        "max_gradient_accumulation_steps": 4,
        "batch_size": 1,
        "total_steps": 600,
        "warmup_steps": 40,
        "log_probs_batch_size": 1,
        "full_last_epoch": True,
    }

    vllm_server_args = {
        # "host": "0.0.0.0",
        # "port": 8000,
        # "served_model_name": model_name,
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
        no_completions=no_completions,  # TODO: Change this to completions_per_prompt
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
        vllm_server_args=vllm_server_args,
        reuse_completions=reuse_completions,
        reuse_completions_batch_size=reuse_completions_batch_size,
    )

    # Calculate and print the total execution time
    total_execution_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(
        f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)"
    )
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
        if "total_start_time" in globals():
            total_execution_time = time.time() - total_start_time
            print(f"\n{'='*50}")
            print(
                f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)"
            )
            print(f"{'='*50}")

        print("Emergency cleanup: Ensuring all vLLM processes are terminated...")
        # Try multiple times to ensure cleanup
        for i in range(5):
            kill_vllm_workers()
            time.sleep(1)  # Give some time between kill attempts

            # Try killing by port in a wider range
            # VLLM could be using ports in a wide range, so scan from 8000-65000
            for port_range_start in range(8000, 63000, 2000):
                port_range_end = min(port_range_start + 2000, 65000)
                try:
                    # Use a range to kill processes more efficiently
                    command = f"for port in $(seq {port_range_start} {port_range_end}); do lsof -ti:$port | xargs kill -9 2>/dev/null || true; done"
                    subprocess.run(
                        command,
                        shell=True,
                        check=False,
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                    )
                except:
                    pass
            time.sleep(0.5)

            # Additional cleanup: find and kill any python processes related to vLLM
            try:
                # Look for vLLM Python processes and kill them
                subprocess.run(
                    "pkill -f 'vllm.entrypoints.openai.api_server' || true",
                    shell=True,
                    check=False,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                )

                # Kill any hanging Python multiprocessing workers
                subprocess.run(
                    "pkill -f 'multiprocessing.spawn import spawn_main' || true",
                    shell=True,
                    check=False,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
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
