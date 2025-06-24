#!/usr/bin/env python
import argparse
import asyncio
import gc
import random
import time
from collections import defaultdict
from functools import partial
from itertools import cycle
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

import torch
from datasets import load_dataset
from omegaconf import OmegaConf

import wandb
from huggingface_hub import HfApi
from branch_reasoning.config import BranchGRPOConfig
from branch_reasoning.countdown_task import transform_countdown_data
from branch_reasoning.generation.completions import generate_completions
from branch_reasoning.generation.vllm_completions import vllm_generate_completions
from branch_reasoning.generation.vllm_generation import kill_vllm_workers
from branch_reasoning.log_probs import populate_log_probs
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
from branch_reasoning.training import get_optimizer_and_scheduler, train_grpo
from branch_reasoning.utils.utils import _print_gpu_memory, move_optimizer_state
from branch_reasoning.prompt_completion_dataset import (
    PromptCompletionDataset,
    collate_fn,
)
from branch_reasoning.utils.utils import find_largest_completion


async def main():
    parser = argparse.ArgumentParser(description="Train a model using Branch GRPO")
    parser.add_argument(
        "--config_path", type=str, help="Path to the configuration file"
    )
    args = parser.parse_args()

    config = OmegaConf.structured(BranchGRPOConfig())
    if args.config_path:
        yaml_config = OmegaConf.load(args.config_path)
        config = OmegaConf.merge(config, yaml_config)

    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)

    print(config)

    model, reference_model, tokenizer = get_models_and_tokenizers(
        config.model.model_name,
        config.model.reference_model_name,
        config.model.tokenizer_name,
        config.model.use_bfloat16,
        beta=config.training.beta,
    )
    tokenizer.save_pretrained(config.experiment.model_dir)
    if reference_model is not None:
        reference_model = reference_model.to("cpu")
    model = model.to("cpu")

    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        config.training,
        config.experiment.iterations,
        config.generation.no_completions,
    )

    task_dataset = load_dataset(config.experiment.hf_dataset_name)["train"]
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
            scoring_variables=OmegaConf.to_container(config.scoring),
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
            scoring_variables=OmegaConf.to_container(config.scoring),
        )

    if config.experiment.wandb_logging:
        wandb.init(
            project=config.experiment.wandb_project_name,
            name=config.experiment.run_name,
            config=config,
        )

    if config.experiment.reuse_completions:
        reuse_completions_dataset = ReuseCompletionsDataset(
            max_length=config.generation.total_completions * 12
        )

    for i in range(config.experiment.iterations):
        start_iteration_time = time.time()
        wandb_logs = defaultdict(float)
        prompt_completions = []
        _print_gpu_memory()
        print(f"Pre Generation, iteration {i}")
        if config.generation.use_vllm:
            try:
                print(
                    f"Generating completions with vLLM, model_name = {config.model.model_name}"
                )
                # Save the model in model_dir
                vllm_server_args = OmegaConf.to_container(config.vllm_server)
                assert isinstance(vllm_server_args, dict)
                model.save_pretrained(config.experiment.model_dir)
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
            except Exception as vllm_error:
                print(f"Error in vLLM generation: {vllm_error}")
                kill_vllm_workers()
                time.sleep(10)
            finally:
                kill_vllm_workers()
                time.sleep(2)
                _print_gpu_memory()
        else:
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

        prompt_completions = populate_log_probs(
            prompt_completions=prompt_completions,
            model=model,
            tokenizer=tokenizer,
            reference_model=reference_model,
            batch_size=config.training.log_probs_batch_size,
        )
        end_log_probs_time = time.time()
        _print_gpu_memory()
        prompt_dataset = PromptCompletionDataset(
            tokenizer=tokenizer,
            prompt_completions=prompt_completions,
        )
        dataloader = DataLoader(
            prompt_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )
        if reference_model is not None:
            reference_model = reference_model.to("cpu")
        del prompt_completions
        _print_gpu_memory()
        torch.cuda.empty_cache()
        move_optimizer_state(optimizer, config.model.device)
        train_grpo(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.model.device,
            iteration=i,
            use_wandb=config.experiment.wandb_logging,
            temperature=config.generation.temperature,
            training_args=config.training,
        )
        end_train_time = time.time()

        # Save model to HuggingFace every save_interval iterations
        if (i + 1) % config.experiment.save_interval == 0:
            print(f"Saving model to HuggingFace repo: {config.experiment.hf_repo_name}")
            try:
                model.push_to_hub(config.experiment.hf_repo_name)
                tokenizer.push_to_hub(config.experiment.hf_repo_name)
                print(f"Successfully saved model to HuggingFace at iteration {i + 1}")
            except Exception as e:
                print(f"Error saving model to HuggingFace: {e}")

        model = model.to("cpu")
        torch.cuda.empty_cache()
        del prompt_dataset
        move_optimizer_state(optimizer, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        _print_gpu_memory()
        print(f"Post Train, iteration {i}")
        if config.experiment.wandb_logging:
            wandb.log(
                {
                    "generation_time": end_generation_time - start_iteration_time,
                    "log_probs_time": end_log_probs_time - end_generation_time,
                    "train_time": end_train_time - end_log_probs_time,
                    "total_iteration_time": end_train_time - start_iteration_time,
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
