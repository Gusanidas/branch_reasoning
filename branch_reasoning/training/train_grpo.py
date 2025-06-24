from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

import wandb


def is_update_step(
    step: int, gradient_accumulation_steps: int, final_steps: int
) -> bool:
    return (step + 1) % gradient_accumulation_steps == 0 or step == final_steps - 1


def train_grpo(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR] = None,
    device: Optional[str] = None,
    iteration: int = 0,
    use_wandb: bool = True,
    temperature: float = 1.0,
    training_args: Dict[str, Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a model using GRPO.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    total_steps = 0
    final_steps = len(dataloader) * num_epochs
    updates = 0
    max_tokens = 0
    max_abs_loss = 0
    threshold_skipped = 0
    accumulated_metrics = defaultdict(list)

    for epoch in range(num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        max_advantage = 0
        min_advantage = 0

        dataloader_iter = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(dataloader_iter):
            total_steps += 1

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            masks = batch["completion_masks"]
            prompt_ids = batch["prompt_ids"]
            completion_ids = batch["completion_ids"]
            completion_masks = batch["completion_masks"]

            advantage = batch["scores"]

            # Get the advantage/score statistics
            max_advantage = max(max_advantage, advantage.max().item())
            min_advantage = min(min_advantage, advantage.min().item())
            avg_abs_advantage = torch.abs(advantage).mean().item()

            # Get the stored log probabilities
            log_probs = batch["log_probs"]
            ref_log_probs = (
                batch["ref_log_probs"] if batch["ref_log_probs"] is not None else None
            )

            # Get model outputs
            full_ids = completion_ids
            max_tokens = max(max_tokens, full_ids.shape[1])
            outputs = model(full_ids, return_dict=True)
            logits = outputs.logits / temperature

            # Slice logits to match the completion part only
            # logits = logits[:, prompt_ids.shape[1]-1:-1, :]
            logits = logits[:, :-1, :]

            # Calculate log probabilities
            model_log_probs = F.log_softmax(logits, dim=-1)
            if total_steps % 200 == 1 and verbose:
                print(f"Shape of model_log_probs = {model_log_probs.shape}")
                print(f"Shape of log_probs = {log_probs.shape}")
                print(f"Shape of ref_log_probs = {ref_log_probs.shape}")
                print(f"Shape of logits = {logits.shape}")
                print(f"Shape of masks = {masks.shape}")
                print(f"max_tokens = {max_tokens}")
                print("--------------------------------")
            # Gather the log probabilities for the actual tokens
            target_tokens = completion_ids[:, 1:]
            model_log_probs = model_log_probs.gather(
                dim=-1, index=target_tokens.unsqueeze(-1)
            )
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
            c_l = torch.sum((ratio < (1 - epsilon_low))).item()
            c_h = torch.sum((ratio > (1 + epsilon_high))).item()
            clipped_total = clipped_low + clipped_high
            if total_steps % 200 == 1 and verbose:
                print(f"c_l = {c_l}, c_h = {c_h}")
                print(f"clipped_low = {clipped_low}, clipped_high = {clipped_high}")
                print(f"epoch = {epoch}, step = {total_steps}")
                print(f"Shape of logits = {logits.shape}")
                print(f"Shape of model_log_probs = {model_log_probs.shape}")
                print(f"Shape of log_probs = {log_probs.shape}")
                print(f"Shape of ref_log_probs = {ref_log_probs.shape}")
                print(f"Shape of target_tokens = {target_tokens.shape}")
                print(f"Shape of masks = {masks.shape}")
                print(f"Shape of advantage = {advantage.shape}")
                print(f"Shape of ratio = {ratio.shape}")
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
                # print(1/0)
            # Clip the ratio
            clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)

            # Calculate percentages
            pct_clipped_low = (
                (clipped_low / total_values) * 100 if total_values > 0 else 0
            )
            pct_clipped_high = (
                (clipped_high / total_values) * 100 if total_values > 0 else 0
            )
            pct_clipped_total = (
                (clipped_total / total_values) * 100 if total_values > 0 else 0
            )

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
                log_ratio = torch.clamp(
                    log_ratio, -clip_log_ratio_value, clip_log_ratio_value
                )
                per_token_kl = torch.exp(log_ratio) - (log_ratio + 1)
                loss_ratio = policy_loss.sum() / (1e-6 + beta * per_token_kl.sum())
                policy_loss += beta * per_token_kl
                kl_loss = per_token_kl * masks
                kl_loss_sum = kl_loss.sum() / mask_sum
                accumulated_metrics["train/kl_loss"].append(kl_loss_sum.item())

            # Apply mask and calculate final loss
            masked_policy_loss = policy_loss * masks

            # Scale the loss
            loss = masked_policy_loss.sum() / (mask_sum + 1e-8)
            loss = loss / gradient_accumulation_steps

            # Accumulate loss
            loss.backward()
            max_abs_loss = max(
                max_abs_loss, torch.abs(loss).item() * gradient_accumulation_steps
            )

            # Log stats
            accumulated_metrics["train/pct_clipped_total"].append(pct_clipped_total)
            accumulated_metrics["train/pct_clipped_low"].append(pct_clipped_low)
            accumulated_metrics["train/pct_clipped_high"].append(pct_clipped_high)
            accumulated_metrics["train/loss"].append(
                loss.item() * gradient_accumulation_steps
            )
            accumulated_metrics["train/avg_abs_advantage"].append(avg_abs_advantage)

            if is_update_step(total_steps, gradient_accumulation_steps, final_steps):
                updates += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

                if grad_norm > max_grad_threshold:
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

                # Log learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                accumulated_metrics["train/learning_rate"].append(current_lr)

                # Log policy loss and grad norm
                accumulated_metrics["train/grad_norm"].append(grad_norm.item())

        if use_wandb:
            avg_metrics = {
                metric: sum(values) / len(values) if values else 0
                for metric, values in accumulated_metrics.items()
            }
            avg_metrics.update(
                {
                    "train/updates": updates,
                    "train/threshold_skipped": threshold_skipped,
                    "train/epoch": epoch,
                    "train/loss_std": (
                        torch.std(
                            torch.tensor(accumulated_metrics["train/loss"])
                        ).item()
                        if accumulated_metrics["train/loss"]
                        else 0
                    ),
                    "train/max_abs_loss": max_abs_loss,
                    "train/max_tokens": max_tokens,
                    "train/max_advantage": max_advantage,
                    "train/min_advantage": min_advantage,
                }
            )
            wandb.log(avg_metrics, step=iteration)

    return accumulated_metrics
