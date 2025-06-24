#!/usr/bin/env python
import argparse
import asyncio
import gc
import time
import os
from functools import partial
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from branch_reasoning.config import BranchGRPOConfig
from branch_reasoning.log_probs import populate_log_probs
from branch_reasoning.training import (
    get_optimizer_and_scheduler,
    save_optimizer_and_scheduler,
    train_with_grpo_distributed,
)
from branch_reasoning.utils.utils import _print_gpu_memory
from branch_reasoning.prompt_completion_dataset import (
    PromptCompletionDataset,
    collate_fn,
)
from branch_reasoning.prompt_completion_dataset import JsonlDataset
from branch_reasoning.utils.utils import move_optimizer_state
import torch.distributed as dist


def is_distributed_environment() -> bool:
    """Check if we're running in a distributed environment (with torchrun)"""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> Tuple[int, int]:
    """Initializes the distributed process group if in distributed environment."""
    if is_distributed_environment():
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        # Single process mode
        return 0, 1


def cleanup_distributed():
    """Cleans up the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int) -> str:
    """Get the appropriate device for the given rank."""
    if torch.cuda.is_available():
        return f"cuda:{rank}"
    else:
        return "cpu"


def wrap_model_for_training(model, rank: int, world_size: int, device: str):
    """Wrap model with DDP if in distributed mode, otherwise return as-is."""
    if world_size > 1 and torch.cuda.is_available():
        return DDP(
            model,
            device_ids=[rank],
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
    else:
        return model


def distribute_work(items: List, rank: int, world_size: int) -> List:
    """Distribute work items across processes"""
    if world_size == 1:
        return items

    items_per_process = len(items) // world_size
    remainder = len(items) % world_size

    # Calculate start and end indices for this rank
    if rank < remainder:
        start_idx = rank * (items_per_process + 1)
        end_idx = start_idx + items_per_process + 1
    else:
        start_idx = rank * items_per_process + remainder
        end_idx = start_idx + items_per_process

    return items[start_idx:end_idx]


def gather_processed_data(local_data, world_size):
    """Gather processed data from all processes using all_gather_object"""
    if world_size == 1 or not dist.is_initialized():
        return local_data

    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, local_data)

    # Flatten the list of lists into a single list
    all_processed_data = []
    for data in gathered_data:
        if data is not None:
            all_processed_data.extend(data)

    return all_processed_data


def create_dataloader(
    dataset, batch_size: int, rank: int, world_size: int, tokenizer, target_length: int
):
    """Create DataLoader with appropriate sampler based on distributed mode."""
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                collate_fn, tokenizer=tokenizer, target_length=target_length
            ),
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_fn, tokenizer=tokenizer, target_length=target_length
            ),
        )


def safe_barrier():
    """Call dist.barrier() only if distributed is initialized."""
    if dist.is_initialized():
        dist.barrier()


async def main():
    config = OmegaConf.structured(BranchGRPOConfig())
    # config = OmegaConf.load("config.yaml")

    # Add from cli
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)
    print(f"config iters = {config.iteration}")

    rank, world_size = setup_distributed()

    # Print mode information
    if rank == 0:
        if world_size > 1:
            print(f"Running in distributed mode with {world_size} processes")
        else:
            print("Running in single process mode")

    try:
        device = get_device(rank)
        print(f"Rank {rank} using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            config.experiment.model_dir, torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.experiment.model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Wrap model for distributed training if needed
        model = wrap_model_for_training(model, rank, world_size, device)

        t0 = time.time()
        if config.training.use_gradient_checkpointing:
            print(f"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--")
            print(
                f"Enabling gradient checkpointing for model optimization (rank {rank})"
            )
            # Handle both DDP and non-DDP models
            model_to_checkpoint = model.module if hasattr(model, "module") else model
            model_to_checkpoint.gradient_checkpointing_enable()
            print(f"Time taken to enable gradient checkpointing = {time.time() - t0}")
            print(f"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--")

        if config.training.use_torch_compile:
            print(f"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--")
            print(f"Enabling torch.compile for model optimization (rank {rank})")
            # model = torch.compile(model, mode="reduce-overhead")
            model = torch.compile(model, mode="default")
            print(f"Time taken to compile model = {time.time() - t0}")

        reference_model = None
        if config.experiment.ref_model_dir is not None:
            reference_model = AutoModelForCausalLM.from_pretrained(
                config.experiment.ref_model_dir, torch_dtype=torch.bfloat16
            ).to(device)
            # Wrap reference model for distributed if needed
            reference_model = wrap_model_for_training(
                reference_model, rank, world_size, device
            )
            reference_model.eval()

        jsonl_dataset = JsonlDataset(config.experiment.data_path)
        if rank == 0:
            print(f"Loaded {len(jsonl_dataset)} completions")

        prompt_completions = []
        for pc in jsonl_dataset:
            prompt_completions.append(pc)

        local_prompt_completions = distribute_work(prompt_completions, rank, world_size)
        print(f"Size of prompt_completions = {len(prompt_completions)}, rank = {rank}")
        print(
            f"Size of local_prompt_completions = {len(local_prompt_completions)}, rank = {rank}"
        )

        local_prompt_completions = populate_log_probs(
            prompt_completions=local_prompt_completions,
            model=model,
            tokenizer=tokenizer,
            reference_model=reference_model if reference_model else None,
            batch_size=config.training.log_probs_batch_size,
        )

        del reference_model
        gc.collect()
        torch.cuda.empty_cache()

        print(
            f"Len of local_prompt_completions = {len(local_prompt_completions)}, rank = {rank}"
        )

        # Gather data from all processes if distributed
        if world_size > 1:
            print("Gathering processed data")
            prompt_completions = gather_processed_data(
                local_prompt_completions, world_size
            )
        else:
            prompt_completions = local_prompt_completions

        print(f"Size of prompt_completions = {len(prompt_completions)}, rank = {rank}")
        print(
            f"Size of local_prompt_completions = {len(local_prompt_completions)}, rank = {rank}"
        )

        safe_barrier()

        if rank == 0:
            print("After log probs")
            _print_gpu_memory()

        prompt_dataset = PromptCompletionDataset(
            prompt_completions=prompt_completions,
            tokenizer=tokenizer,
        )
        print(f"Size of prompt_dataset = {len(prompt_dataset)}")

        dataloader = create_dataloader(
            prompt_dataset,
            config.training.batch_size,
            rank,
            world_size,
            tokenizer,
            config.generation.max_len,
        )

        print(f"Size of prompt_dataset = {len(prompt_dataset)}")
        print(f"config.training.batch_size = {config.training.batch_size}")
        print(f"Size of dataloader = {len(dataloader)}")

        optimizer, scheduler = get_optimizer_and_scheduler(
            model,
            config.training,
            config.experiment.iterations,
            config.generation.no_completions,
            from_checkpoint=True,
            optimizer_path=config.training.optimizer_path,
            scheduler_path=config.training.scheduler_path,
        )
        move_optimizer_state(optimizer, device)
        safe_barrier()

        # print gpu memory
        if rank == 0:
            print("Before training")
            _print_gpu_memory()

        wandb_stats = train_with_grpo_distributed(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            iteration=config.iteration,
            use_wandb=False,
            temperature=config.generation.temperature,
            training_args=config.training,
            rank=rank,
            world_size=world_size,
            training_log_path=config.experiment.training_log_path,
        )

        if rank == 0:
            for key, value in wandb_stats.items():
                print(f"{key}, type = {type(value)}")
            with open(config.experiment.training_log_path, "w") as f:
                json.dump(wandb_stats, f)
                f.write("\n")

        safe_barrier()

        save_optimizer_and_scheduler(
            optimizer,
            scheduler,
            config.training.optimizer_path,
            config.training.scheduler_path,
        )

        # Save model - handle both DDP and non-DDP cases
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(config.experiment.model_dir)

    except Exception as e:
        print(f"An error occurred during training (rank {rank}): {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Ensure distributed cleanup happens regardless of success or failure
        cleanup_distributed()


if __name__ == "__main__":
    asyncio.run(main())
