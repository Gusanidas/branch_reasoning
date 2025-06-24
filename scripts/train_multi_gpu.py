#!/usr/bin/env python
import asyncio
import gc
import random
import subprocess
import time
from collections import defaultdict
from functools import partial
from itertools import cycle
from typing import Any, Dict, List, Optional
import json

import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

import wandb
from huggingface_hub import HfApi
from branch_reasoning.config import BranchGRPOConfig
from branch_reasoning.countdown_task import transform_countdown_data
from branch_reasoning.generation.completions import generate_completions
from branch_reasoning.generation.vllm_completions import vllm_generate_completions
from branch_reasoning.generation.vllm_generation import kill_vllm_workers
from branch_reasoning.models.model_loader import get_models_and_tokenizers
from branch_reasoning.prompts import base_prompt, get_format_and_examples
from branch_reasoning.reuse_completions import ReuseCompletionsDataset
from branch_reasoning.scoring import (
    CompletionScorer,
    match_format_approximately,
    match_format_exactly,
    match_format_loosely,
    rate_countdown_answer,
    rate_countdown_answer_individual,
    score_branch_format,
    score_branch_format_approx,
    score_branch_format_loose,
    score_by_length,
)
from branch_reasoning.training import (
    get_optimizer_and_scheduler,
    save_optimizer_and_scheduler,
)
from branch_reasoning.utils.utils import _print_gpu_memory
from branch_reasoning.prompt_completion_dataset import serialize_prompt_completion
from branch_reasoning.utils.utils import find_largest_completion


def launch_training(i: int):
    """
    Launches the training script using torchrun.
    """
    print("Launching training script with torchrun...")
    try:
        command = [
            "torchrun",
            "--nproc_per_node=2",
            "_train_multi_gpu_batch.py",
            f"iteration={i}",
        ]

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        print("Training script finished successfully.")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

    except FileNotFoundError:
        print(
            "Error: 'torchrun' command not found. Make sure PyTorch is installed and in your PATH."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing training script: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


async def main():

    default_cfg = OmegaConf.structured(BranchGRPOConfig())
    cli_args = OmegaConf.from_cli()
    try:
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
        config = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    except Exception as e:
        print(f"Error loading config file: {e}")
        del cli_args.config
        config = OmegaConf.merge(default_cfg, cli_args)

    print(f"config = {config}")

    model, reference_model, tokenizer = get_models_and_tokenizers(
        config.model.model_name,
        config.model.reference_model_name,
        config.model.tokenizer_name,
        config.model.use_bfloat16,
        beta=config.training.beta,
    )
    if config.experiment.model_dir is not None:
        model.save_pretrained(config.experiment.model_dir)
        tokenizer.save_pretrained(config.experiment.model_dir)
    if config.experiment.ref_model_dir is not None and reference_model is not None:
        reference_model.save_pretrained(config.experiment.ref_model_dir)
        tokenizer.save_pretrained(config.experiment.ref_model_dir)

    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        config.training,
        config.experiment.iterations,
        config.generation.no_completions,
    )
    save_optimizer_and_scheduler(
        optimizer,
        scheduler,
        config.training.optimizer_path,
        config.training.scheduler_path,
    )

    task_dataset = load_dataset(config.experiment.hf_dataset_name)["train"]
    assert isinstance(task_dataset, Dataset)
    format_prompt, examples = get_format_and_examples(
        config.generation.branch_completions,
        config.generation.max_branching_points,
        config.generation.branching_factor,
    )
    examples = [""]
    if config.generation.use_examples:
        examples = ["Example:\n" + example for example in examples]
        examples = examples + [""] * len(examples)

    dataset = transform_countdown_data(
        task_dataset,
        base_prompt_template=base_prompt,
        template_func=None,
        format_prompt=format_prompt,
        examples=examples,
    )
    dataset = dataset.shuffle()
    cycling_dataset = cycle(dataset)

    pscore_rate_countdown_answer_individual = partial(
        rate_countdown_answer_individual,
        normalize_all_branches=config.scoring.normalize_all_branches,
    )
    pscore_rate_countdown_answer_individual.__name__ = (
        "rate_countdown_answer_individual"
    )
    pscore_branch_format_approx = partial(
        score_branch_format_approx,
        max_branches=config.generation.max_branching_points,
        branch_factor=config.generation.branching_factor,
    )
    pscore_branch_format_loose = partial(
        score_branch_format_loose,
        max_branches=config.generation.max_branching_points,
        branch_factor=config.generation.branching_factor,
    )
    pscore_branch_format = partial(
        score_branch_format,
        max_branches=config.generation.max_branching_points,
        branch_factor=config.generation.branching_factor,
    )

    pscore_branch_format_approx.__name__ = "score_branch_format_approx"
    pscore_branch_format_loose.__name__ = "score_branch_format_loose"
    pscore_branch_format.__name__ = "score_branch_format"

    if config.generation.branch_completions:
        completion_scorer = CompletionScorer(
            scoring_functions=[
                match_format_exactly,
                rate_countdown_answer,
                match_format_approximately,
                match_format_loosely,
                pscore_branch_format_approx,
                pscore_branch_format_loose,
                pscore_branch_format,
                pscore_rate_countdown_answer_individual,
                score_by_length,
            ],
            scoring_variables=config.scoring,
        )
    else:
        completion_scorer = CompletionScorer(
            scoring_functions=[
                match_format_exactly,
                rate_countdown_answer,
                match_format_approximately,
                match_format_loosely,
                score_by_length,
            ],
            scoring_variables=config.scoring,
        )
    if config.experiment.wandb_logging:
        wandb.init(
            project=config.experiment.wandb_project_name,
            name=config.experiment.run_name,
            id=config.experiment.run_name + "_batch",
            config=config,
        )
        # resume="allow",
        # settings=wandb.Settings(
        #    x_label="rank_0",
        #    mode="shared",
        #    x_primary=True,
        #    )
        # )

    if config.experiment.reuse_completions:
        reuse_completions_dataset = ReuseCompletionsDataset(
            max_length=config.generation.total_completions * 12
        )

    del model, reference_model
    torch.cuda.empty_cache()
    gc.collect()
    for i in range(config.experiment.iterations):
        print(f"Iteration {i}")
        start_iteration_time = time.time()

        wandb_logs = defaultdict(float)
        prompt_completions = []
        _print_gpu_memory()
        print(f"Pre Generation, iteration {i}")
        if config.generation.use_vllm:
            # Retry vllm generation up to 3 times with increasing wait times
            max_retries = 3
            wait_times = [
                0,
                5,
                20,
            ]  # No wait on first attempt, 5s after first failure, 20s after second

            for attempt in range(max_retries):
                try:
                    print(
                        f"Generating completions with vLLM, model_name = {config.model.model_name} (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt > 0:
                        print(f"Waiting {wait_times[attempt]} seconds before retry...")
                        time.sleep(wait_times[attempt])

                    vllm_server_args = OmegaConf.to_container(config.vllm_server)
                    assert isinstance(vllm_server_args, dict)
                    prompt_completions = await vllm_generate_completions(
                        tokenizer=tokenizer,
                        dataset=cycling_dataset,
                        total_completions=config.generation.total_completions,
                        completions_per_prompt=config.generation.no_completions,
                        gen_batch_size=config.generation.gen_batch_size,
                        current_iter=i,
                        max_len=config.generation.max_len,
                        model_dir=config.experiment.model_dir,
                        vllm_server_args=vllm_server_args,
                        log_file=f"./logs/vllm_train_iter_{i}.log",
                        generation_args=config.generation.opt_generation_args,
                        wandb_logging=config.experiment.wandb_logging,
                        branch_completions=config.generation.branch_completions,
                        branching_factor=config.generation.branching_factor,
                        max_branching_points=config.generation.max_branching_points,
                        temperature=config.generation.temperature,
                    )
                    # If successful, break out of the retry loop
                    break
                except Exception as vllm_error:
                    print(
                        f"Error in vLLM generation (attempt {attempt + 1}/{max_retries}): {vllm_error}"
                    )
                    kill_vllm_workers()
                    if attempt == max_retries - 1:
                        print(
                            "All vLLM generation attempts failed. Continuing with empty completions."
                        )
                        prompt_completions = []
                finally:
                    kill_vllm_workers()
                    time.sleep(2)
                    _print_gpu_memory()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.experiment.model_dir, torch_dtype=torch.bfloat16
            ).to(config.model.device)
            prompt_completions = generate_completions(
                model=model,
                tokenizer=tokenizer,
                dataset=cycling_dataset,
                total_completions=config.generation.total_completions,
                completions_per_prompt=config.generation.no_completions,
                gen_batch_size=config.generation.gen_batch_size,
                current_iter=i,
                max_len=config.generation.max_len,
                generation_args=config.generation.opt_generation_args,
                device=config.model.device,
                wandb_logging=config.experiment.wandb_logging,
                branch_completions=config.generation.branch_completions,
                branching_factor=config.generation.branching_factor,
                max_branching_points=config.generation.max_branching_points,
                temperature=config.generation.temperature,
            )
        end_generation_time = time.time()
        if len(prompt_completions) == 0:
            print(f"No completions generated, skipping iteration {i}")
            continue

        if config.experiment.reuse_completions:
            if i > config.experiment.reuse_completions_start_iter:
                reuse_completions_batch = reuse_completions_dataset.next_batch(
                    config.experiment.reuse_completions_batch_size,
                    recent_batch_size=config.experiment.reuse_completions_recent_batch_size,
                )
            reuse_completions_dataset.add_completions(prompt_completions)
            if (
                i > config.experiment.reuse_completions_start_iter
                and reuse_completions_batch is not None
                and len(reuse_completions_batch) > 0
            ):
                prompt_completions = prompt_completions + reuse_completions_batch
                random.shuffle(prompt_completions)

        prompt_completions = completion_scorer.score_completions(
            prompt_completion_list=prompt_completions,
            wandb_logs=wandb_logs,
        )

        # Write prompt_completions to jsonl file
        print("Serializing completions...")
        serialized_completions = [
            serialize_prompt_completion(pc) for pc in prompt_completions
        ]

        output_file_path = config.experiment.data_path
        with open(output_file_path, "w", encoding="utf-8") as f:
            for completion in serialized_completions:
                f.write(json.dumps(completion) + "\n")

        print(
            f"Successfully saved {len(serialized_completions)} serialized completions to {output_file_path}"
        )

        print(
            f"Launching training, iteration {i}, time {time.time()-start_iteration_time:.2f}s"
        )
        try:
            launch_training(i)
            end_training_time = time.time()
            print(
                f"Training finished, iteration {i}, time {end_training_time-start_iteration_time:.2f}s"
            )
        except Exception as training_error:
            print(f"Error during training for iteration {i}: {training_error}")
            end_training_time = time.time()
            print(
                f"Training failed, iteration {i}, time {end_training_time-start_iteration_time:.2f}s"
            )

        # Save model to HuggingFace every save_interval iterations
        if (i + 1) % config.experiment.save_interval == 0:
            print(f"Saving model to HuggingFace repo: {config.experiment.hf_repo_name}")
            try:
                # Load the trained model from checkpoint
                trained_model = AutoModelForCausalLM.from_pretrained(
                    config.experiment.model_dir, torch_dtype=torch.bfloat16
                )
                trained_model.push_to_hub(config.experiment.hf_repo_name)
                tokenizer.push_to_hub(config.experiment.hf_repo_name)
                print(f"Successfully saved model to HuggingFace at iteration {i + 1}")
                del trained_model  # Clean up memory
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error saving model to HuggingFace: {e}")

        # Read training metrics from file and log to wandb
        if config.experiment.wandb_logging and config.experiment.training_log_path:
            try:
                with open(
                    config.experiment.training_log_path, "r", encoding="utf-8"
                ) as f:
                    lines = f.readlines()
                    print(f"Lines = {lines}")
                    if lines:
                        last_line = lines[-1].strip()
                        training_metrics = json.loads(last_line)

                        training_metrics_copy = training_metrics.copy()
                        training_metrics_copy.pop("iteration", None)
                        wandb.log(training_metrics_copy, step=i)
                        print(f"Logged training metrics to wandb for iteration {i}")

            except Exception as e:
                print(
                    f"Error reading training metrics from {config.experiment.training_log_path}: {e}"
                )
        if config.experiment.wandb_logging:
            wandb.log(
                {
                    "generation_time": end_generation_time - start_iteration_time,
                    "log_prob_and_train_time": end_training_time - end_generation_time,
                    "total_iteration_time": end_training_time - start_iteration_time,
                },
                step=i,
            )
            for key, value in wandb_logs.items():
                if key + "_steps" in wandb_logs:
                    v = value / wandb_logs[key + "_steps"]
                    wandb.log({key: v}, step=i)
                elif not key.endswith("_steps"):
                    wandb.log({key: value}, step=i)
    if config.experiment.wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
