import torch
import random
import time
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import wandb
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR

from branch_reasoning.generation.completions import PromptCompletion, Branch

def train_with_grpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_completions: List[PromptCompletion],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR] = None,
    device: Optional[str] = None,
    iteration: int = 0,
    use_wandb: bool = True,
    temperature: float = 1.0,
    training_args: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Train a model using the GRPO (Generalized Reinforcement Learning from Policy Optimization) algorithm.

    Args:
        model: HuggingFace model to train
        tokenizer: HuggingFace tokenizer
        prompt_completions: List of PromptCompletion objects
        optimizer: Optimizer to use for training
        scheduler: Learning rate scheduler
        gradient_accumulation_steps: Number of prompts per batch (each prompt has several completions)
        device: Device to run training on ('cuda' or 'cpu')
        iteration: Current iteration number for logging
        use_wandb: Whether to use wandb for logging
        training_args: Dictionary containing training arguments

    Returns:
        Dict containing training statistics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.train()

    # Extract training parameters
    num_epochs = training_args.get("num_epochs", 1)
    epsilon_low = training_args.get("epsilon_low", 0.2)
    epsilon_high = training_args.get("epsilon_high", 0.2)
    max_grad_norm = training_args.get("max_grad_norm", 1.0)
    beta = training_args.get("beta", 0.0)
    gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 1)
    
    # Initialize statistics
    stats = {
        'epoch_losses': [],
        'policy_losses': []
    }

    # Group prompt_completions for batching
    total_prompts = len(prompt_completions) * num_epochs
    t0 = time.time()
    print(f"----------0----------")
    print(f"Inside train_with_grpo, total_prompts = {total_prompts}, temperature = {temperature}")
    prompt_steps = 0
    total_steps = 0
    skipped_branches = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        accumulated_metrics = defaultdict(list)
        
        max_advantage = 0
        min_advantage = 0
        
        for prompt_completion in prompt_completions:
            prompt_steps += 1
            completions_per_prompt = len(prompt_completion.branched_completions)
            for branched_completion in prompt_completion.branched_completions:
                # Process each branch in the branched completion
                for branch in branched_completion.branches:
                    total_steps += 1
                    if branch.score is None or branch.log_probs is None:
                        skipped_branches += 1
                        print(f"branch.score is None or branch.log_probs is None")
                        continue
                    
                    # Encode the prompt and completion
                    prompt_ids = tokenizer(prompt_completion.prompt, return_tensors="pt").input_ids.to(device)
                    completion_ids = tokenizer(branch.completion, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    
                    # Get the advantage/score
                    advantage = torch.tensor(branch.score).to(device)
                    max_advantage = max(max_advantage, advantage)
                    min_advantage = min(min_advantage, advantage)
                    avg_abs_advantage = torch.abs(advantage)
                    
                    # Get the stored log probabilities
                    log_probs = branch.log_probs.to(device)
                    ref_log_probs = branch.ref_log_probs.to(device) if branch.ref_log_probs is not None else None
                    
                    
                    # Get model outputs
                    full_ids = completion_ids
                    outputs = model(full_ids, return_dict=True)
                    logits = outputs.logits / temperature
                    
                    # Slice logits to match the completion part only
                    logits = logits[:, prompt_ids.shape[1]-1:-1, :]
                    
                    # Calculate log probabilities
                    model_log_probs = F.log_softmax(logits, dim=-1)
                    model_log_probs = model_log_probs[0]
                    
                    # Gather the log probabilities for the actual tokens
                    target_tokens = completion_ids[0,prompt_ids.shape[1]:]
                    model_log_probs = model_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1))
                    model_log_probs = model_log_probs.squeeze(-1)
                    
                    # Calculate ratio between new and old policy
                    ratio = torch.exp(model_log_probs - log_probs)
                    masks = torch.ones_like(model_log_probs)
                    
                    # Calculate statistics on clipped values
                    total_values = torch.sum(masks).item()
                    if random.random() < 0.01 or total_steps <2:
                        print(f"Shape of masks = {masks.shape}")
                        print(f"Shape of ratio = {ratio.shape}")
                        print(f"Shape of log_probs = {log_probs.shape}")
                        print(f"Shape of model_log_probs = {model_log_probs.shape}")
                        print(f"Shape of completion_ids = {completion_ids.shape}")
                        print(f"Shape of prompt_ids = {prompt_ids.shape}")
                        print(f"first 5 log_probs: {log_probs[:5]}")
                        print(f"first 5 model_log_probs: {model_log_probs[:5]}")
                        if ref_log_probs is not None:
                            print(f"first 5 ref_log_probs: {ref_log_probs[:5]}")
                    
                    clipped_low = torch.sum((ratio < (1 - epsilon_low)) & masks.bool()).item()
                    clipped_high = torch.sum((ratio > (1 + epsilon_high)) & masks.bool()).item()
                    clipped_total = clipped_low + clipped_high
                    
                    # Clip the ratio
                    clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
                    
                    # Calculate percentages
                    pct_clipped_low = (clipped_low / total_values) * 100 if total_values > 0 else 0
                    pct_clipped_high = (clipped_high / total_values) * 100 if total_values > 0 else 0
                    pct_clipped_total = (clipped_total / total_values) * 100 if total_values > 0 else 0
                    
                    # Calculate policy loss
                    policy_loss_unclipped = ratio * advantage
                    policy_loss_clipped = clipped_ratio * advantage
                    policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped)
                    mask_sum = masks.sum()
                    
                    # Add KL penalty if beta > 0
                    if beta > 0 and ref_log_probs is not None:
                        per_token_kl = (
                            torch.exp(ref_log_probs - model_log_probs) - (ref_log_probs - model_log_probs) - 1
                        )
                        policy_loss += beta * per_token_kl
                        kl_loss = per_token_kl * masks
                        kl_loss_sum = kl_loss.sum() / mask_sum
                        accumulated_metrics["train/kl_loss"].append(kl_loss_sum.item())
                    
                    # Apply mask and calculate final loss
                    masked_policy_loss = policy_loss * masks
                    
                    
                    # Scale the loss (divide by gradient_accumulation_steps * completions_per_prompt)
                    loss = masked_policy_loss.sum() / (mask_sum + 1e-8)
                    loss = loss / (gradient_accumulation_steps * completions_per_prompt)
                    
                    # Accumulate loss
                    loss.backward()
                    
                    
                    # Log stats
                    accumulated_metrics["train/pct_clipped_total"].append(pct_clipped_total)
                    accumulated_metrics["train/pct_clipped_low"].append(pct_clipped_low)
                    accumulated_metrics["train/pct_clipped_high"].append(pct_clipped_high)
                    accumulated_metrics["train/loss"].append(loss.item() * gradient_accumulation_steps * completions_per_prompt)
                    accumulated_metrics["train/avg_abs_advantage"].append(avg_abs_advantage.item())
            
            if prompt_steps % gradient_accumulation_steps == gradient_accumulation_steps-1 or prompt_steps == total_prompts-1:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
            
                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                accumulated_metrics["train/learning_rate"].append(current_lr)
            
                # Log policy loss and grad norm
                accumulated_metrics["train/grad_norm"].append(grad_norm.item())

        random.shuffle(prompt_completions)
        
        # Log epoch metrics to wandb
        if use_wandb:
            # Calculate averages of accumulated metrics
            avg_metrics = {
                metric: sum(values) / len(values) if values else 0
                for metric, values in accumulated_metrics.items()
            }
            avg_metrics.update({
                "train/step": iteration,
                "train/epoch": epoch,
                "train/max_advantage": max_advantage,
                "train/min_advantage": min_advantage,
                "train/loss_std": torch.std(torch.tensor(accumulated_metrics["train/loss"])).item() if accumulated_metrics["train/loss"] else 0,
            })
            wandb.log(avg_metrics, step=iteration)
            
            # Record stats
            avg_epoch_loss = epoch_loss / total_prompts if total_prompts > 0 else 0
            stats['epoch_losses'].append(avg_epoch_loss)
            if accumulated_metrics["train/policy_loss"]:
                avg_policy_loss = sum(accumulated_metrics["train/policy_loss"]) / len(accumulated_metrics["train/policy_loss"])
                stats['policy_losses'].append(avg_policy_loss)
            else:
                stats['policy_losses'].append(0.0)

    print(f"Out of train_with_grpo, skipped_branches = {skipped_branches} out of {total_steps}")
    print(f"Time taken: {time.time() - t0} seconds")

    return stats

    