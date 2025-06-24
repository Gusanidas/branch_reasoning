from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedTokenizer
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch.nn import functional as F
from collections import defaultdict
import wandb
from tqdm import tqdm
import json
import time


def regular_forward_pass(model, full_ids, temperature):
    """Regular forward pass without compilation."""
    outputs = model(full_ids, return_dict=True)
    logits = outputs.logits / temperature
    return logits[:, :-1, :]


@torch.compile(mode="reduce-overhead")
def compiled_forward_pass(model, full_ids, temperature):
    """Compiled forward pass for better performance."""
    outputs = model(full_ids, return_dict=True)
    logits = outputs.logits / temperature
    return logits[:, :-1, :]


def is_update_step(
    step: int, gradient_accumulation_steps: int, final_steps: int
) -> bool:
    return (step + 1) % gradient_accumulation_steps == 0 or step == final_steps - 1


def process_training_step(
    model: DDP,
    batch: Dict[str, Any],
    device: str,
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    beta: float,
    clip_log_ratio_value: float,
    gradient_accumulation_steps: int,
    total_steps: int,
    use_torch_compile: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Process a single training step and return metrics."""
    # Move batch to device
    # for key in batch:
    #    if isinstance(batch[key], torch.Tensor):
    #        batch[key] = batch[key].to(device)

    masks = batch["completion_masks"].to(device)
    completion_ids = batch["completion_ids"].to(device)
    completion_masks = batch["completion_masks"].to(device)

    advantage = batch["scores"].to(device)

    # Get the advantage/score statistics
    max_advantage = advantage.max().item()
    min_advantage = advantage.min().item()
    avg_abs_advantage = torch.abs(advantage).mean().item()

    # Get the stored log probabilities
    log_probs = batch["log_probs"].to(device)
    ref_log_probs = (
        batch["ref_log_probs"].to(device)
        if batch["ref_log_probs"] is not None
        else None
    )

    # Get model outputs - use compiled or regular forward pass based on config
    full_ids = completion_ids
    max_tokens = full_ids.shape[1]
    if use_torch_compile and False:
        logits = compiled_forward_pass(model, full_ids, temperature)
    else:
        logits = regular_forward_pass(model, full_ids, temperature)

    # Calculate log probabilities
    model_log_probs = F.log_softmax(logits, dim=-1)

    if total_steps == 1 and verbose:
        print(f"Shape of model_log_probs = {model_log_probs.shape}")
        print(f"Shape of log_probs = {log_probs.shape}")
        print(f"Shape of ref_log_probs = {ref_log_probs.shape}")
        print(f"Shape of logits = {logits.shape}")
        print(f"Shape of masks = {masks.shape}")
        print("--------------------------------")

    # Gather the log probabilities for the actual tokens
    target_tokens = completion_ids[:, 1:]
    model_log_probs = model_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1))
    model_log_probs = model_log_probs.squeeze(-1)

    # if updates == 0:
    #    log_probs = model_log_probs

    # Calculate ratio between new and old policy
    ratio = torch.exp(model_log_probs - log_probs)
    masks = completion_masks

    # Calculate statistics on clipped values
    total_values = torch.sum(masks).item()
    clipped_low = torch.sum((ratio < (1 - epsilon_low)) & masks.bool()).item()
    clipped_high = torch.sum((ratio > (1 + epsilon_high)) & masks.bool()).item()
    clipped_total = clipped_low + clipped_high

    if total_steps == 1 and verbose:
        print(f"Shape of ratio = {ratio.shape}")
        print(f"Shape of masks = {masks.shape}")
        print(f"Shape of advantage = {advantage.shape}")
        print("--------------------------------")
        for i in range(len(log_probs)):
            print(f"log_probs[{i}][:10] = {log_probs[i][:10]}")
            print(f"model_log_probs[{i}][:10] = {model_log_probs[i][:10]}")
            print(f"ref_log_probs[{i}][:10] = {ref_log_probs[i][:10]}")
            print(f"ratio[{i}][:10] = {ratio[i][:10]}")
            print(f"advantage[{i}] = {advantage[i]}")
            print(f"masks[{i}][:10] = {masks[i][:10]}")
            print("-=--=-=-=-=-=-=-=-=-=-=-")
            print(f"log_probs[{i}][-10:] = {log_probs[i][-10:]}")
            print(f"model_log_probs[{i}][-10:] = {model_log_probs[i][-10:]}")
            print(f"ref_log_probs[{i}][-10:] = {ref_log_probs[i][-10:]}")
            print(f"ratio[{i}][-10:] = {ratio[i][-10:]}")
            print(f"advantage[{i}] = {advantage[i]}")
            print(f"masks[{i}][-10:] = {masks[i][-10:]}")
            print("--------------------------------")

    # Clip the ratio
    clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)

    # Calculate percentages
    pct_clipped_low = (clipped_low / total_values) * 100 if total_values > 0 else 0
    pct_clipped_high = (clipped_high / total_values) * 100 if total_values > 0 else 0
    pct_clipped_total = (clipped_total / total_values) * 100 if total_values > 0 else 0

    # Calculate policy loss
    policy_loss_unclipped = ratio * advantage.unsqueeze(-1)
    policy_loss_clipped = clipped_ratio * advantage.unsqueeze(-1)
    policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped)
    mask_sum = masks.sum()

    # Add KL penalty if beta > 0
    loss_ratio = 1.0
    kl_loss_sum = 0.0
    if beta > 0 and ref_log_probs is not None:
        log_ratio = ref_log_probs - model_log_probs
        log_ratio = torch.clamp(log_ratio, -clip_log_ratio_value, clip_log_ratio_value)
        per_token_kl = torch.exp(log_ratio) - (log_ratio + 1)
        loss_ratio = policy_loss.sum() / (1e-6 + beta * per_token_kl.sum())
        policy_loss += beta * per_token_kl
        kl_loss = per_token_kl * masks
        kl_loss_sum = (kl_loss.sum() / mask_sum).item()

    # Apply mask and calculate final loss
    masked_policy_loss = policy_loss * masks

    # Scale the loss
    loss = masked_policy_loss.sum() / (mask_sum + 1e-8)
    loss = loss / gradient_accumulation_steps

    # Accumulate loss
    loss.backward()

    # Return metrics
    return {
        "loss": loss.item() * gradient_accumulation_steps,
        "pct_clipped_total": pct_clipped_total,
        "pct_clipped_low": pct_clipped_low,
        "pct_clipped_high": pct_clipped_high,
        "avg_abs_advantage": avg_abs_advantage,
        "loss_ratio": loss_ratio,
        "kl_loss": kl_loss_sum,
        "max_advantage": max_advantage,
        "min_advantage": min_advantage,
        "max_tokens": max_tokens,
    }


def train_with_grpo_distributed(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR] = None,
    device: Optional[str] = None,
    iteration: int = 0,
    use_wandb: bool = True,
    temperature: float = 1.0,
    training_args: Optional[Dict[str, Any]] = None,
    rank: int = 0,
    world_size: int = 1,
    training_log_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a model using GRPO with distributed data parallel.
    """
    if device is None:
        device = f"cuda:{rank}"

    model = model.to(device)
    model.train()

    # Extract training parameters
    num_epochs = training_args.get("num_epochs", 1)
    epsilon_low = training_args.get("epsilon_low", 0.2)
    epsilon_high = training_args.get("epsilon_high", 0.2)
    max_grad_norm = training_args.get("max_grad_norm", 1.0)
    beta = training_args.get("beta", 0.0)
    gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 1)
    max_grad_threshold = training_args.get("max_grad_threshold", 100 * max_grad_norm)
    clip_log_ratio_value = training_args.get("clip_log_ratio_value", 100)
    use_torch_compile = training_args.get("use_torch_compile", False)

    # Initialize statistics
    stats = {"epoch_losses": [], "policy_losses": []}

    final_steps = len(dataloader) * num_epochs
    total_steps = 0
    updates = 0
    max_tokens = 0
    max_abs_loss = 0
    threshold_skipped = 0
    shortest_epoch, longest_epoch = 1000000, 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}, rank {rank}")
        t0 = time.time()
        epoch_loss = 0
        optimizer.zero_grad()

        accumulated_metrics = defaultdict(list)
        max_advantage = 0
        min_advantage = 0

        dataloader_iter = dataloader

        steps_per_epoch = 0
        for step, batch in enumerate(dataloader_iter):
            total_steps += 1
            steps_per_epoch += 1

            is_update = is_update_step(step, gradient_accumulation_steps, final_steps)

            if is_update or world_size == 1:
                # Regular training step for update steps
                step_metrics = process_training_step(
                    model,
                    batch,
                    device,
                    temperature,
                    epsilon_low,
                    epsilon_high,
                    beta,
                    clip_log_ratio_value,
                    gradient_accumulation_steps,
                    total_steps,
                    use_torch_compile,
                    verbose,
                )
            else:
                # Use no_sync for non-update steps to avoid synchronization overhead
                with model.no_sync():
                    step_metrics = process_training_step(
                        model,
                        batch,
                        device,
                        temperature,
                        epsilon_low,
                        epsilon_high,
                        beta,
                        clip_log_ratio_value,
                        gradient_accumulation_steps,
                        total_steps,
                        use_torch_compile,
                        verbose,
                    )

            # Update global statistics
            max_tokens = max(max_tokens, step_metrics["max_tokens"])
            max_advantage = max(max_advantage, step_metrics["max_advantage"])
            min_advantage = min(min_advantage, step_metrics["min_advantage"])
            max_abs_loss = max(max_abs_loss, abs(step_metrics["loss"]))

            # Log stats
            accumulated_metrics["train/pct_clipped_total"].append(
                step_metrics["pct_clipped_total"]
            )
            accumulated_metrics["train/pct_clipped_low"].append(
                step_metrics["pct_clipped_low"]
            )
            accumulated_metrics["train/pct_clipped_high"].append(
                step_metrics["pct_clipped_high"]
            )
            accumulated_metrics["train/loss"].append(step_metrics["loss"])
            accumulated_metrics["train/avg_abs_advantage"].append(
                step_metrics["avg_abs_advantage"]
            )
            accumulated_metrics["train/is_update"].append(is_update)
            if step_metrics["kl_loss"] > 0:
                accumulated_metrics["train/kl_loss"].append(step_metrics["kl_loss"])

            if is_update:
                updates += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

                if grad_norm > max_grad_threshold:
                    if rank == 0:
                        print(
                            f"Gradient norm {grad_norm} is greater than max_grad_threshold {max_grad_threshold}"
                        )
                        print(f"Skipping update")
                    threshold_skipped += 1
                    optimizer.zero_grad()
                    accumulated_metrics["train/grad_norm"].append(grad_norm.item())
                    continue

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                # Log learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                accumulated_metrics["train/learning_rate"].append(current_lr)

                # Log policy loss and grad norm
                accumulated_metrics["train/grad_norm"].append(grad_norm.item())

        t1 = time.time()
        print(f"Time taken for epoch {epoch} = {t1 - t0}")
        shortest_epoch = min(shortest_epoch, t1 - t0)
        longest_epoch = max(longest_epoch, t1 - t0)

        # Calculate averages of accumulated metrics
        avg_metrics = {
            metric: sum(values) / len(values) if values else 0
            for metric, values in accumulated_metrics.items()
        }
        for key, value in avg_metrics.items():
            if isinstance(value, torch.Tensor):
                avg_metrics[key] = value.item()
        avg_metrics.update(
            {
                "train/updates": updates,
                "train/threshold_skipped": threshold_skipped,
                "train/epoch": epoch,
                "train/loss_std": (
                    torch.std(torch.tensor(accumulated_metrics["train/loss"])).item()
                    if accumulated_metrics["train/loss"]
                    else 0
                ),
                "train/max_abs_loss": max_abs_loss,
                "train/max_tokens": max_tokens,
                "train/max_advantage": max_advantage,
                "train/min_advantage": min_advantage,
                "train/steps_per_epoch": steps_per_epoch,
                "iteration": iteration,
                "train/shortest_epoch": shortest_epoch,
                "train/longest_epoch": longest_epoch,
            }
        )

    if rank == 0:
        print(
            f"Training completed. Updates: {updates}, Threshold skipped: {threshold_skipped}"
        )

    if rank != 0:
        avg_metrics = {}

    return avg_metrics
