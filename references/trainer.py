import torch
import random
import torch.nn.functional as F
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import wandb
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from filter_top_k import filter_top_k_logits

def train_with_grpo(
    model: PreTrainedModel,
    dataloader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR] = None,
    gradient_accumulation_steps: int = 1,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    max_grad_norm: float = 1.0,
    device: Optional[str] = None,
    iteration: int = 0,
    use_wandb: bool = True,
    update_interval: int = 10,
    beta: float = 0.0,
    tokenizer: PreTrainedTokenizer = None,
    temperature: float = 1.0,
    loss_multiplier: float = 1.0,
    top_k: int = 50,
    log_prob_bias: float = -40.0,
) -> Dict[str, Any]:
    """
    Train a model using the GRPO (Generalized Reinforcement Learning from Policy Optimization) algorithm.

    Args:
        model: HuggingFace model to train
        dataloader: DataLoader containing tokens, masks, and advantages
        num_epochs: Number of training epochs
        optimizer: Optimizer to use for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        epsilon_low: Lower clipping threshold for importance weight
        epsilon_high: Upper clipping threshold for importance weight
        max_grad_norm: Maximum gradient norm for clipping
        device: Device to run training on ('cuda' or 'cpu')
        iteration: Current iteration number for logging

    Returns:
        Dict containing training statistics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.train()

    if gradient_accumulation_steps <1:
        gradient_accumulation_steps = len(dataloader)


    # Prepare data

    stats = {
        'epoch_losses': [],
        'policy_losses': []
    }

    # Initialize storage for old probabilities
    old_probs_dict = {}
    total_step = 0
    total_steps = len(dataloader) * num_epochs
    print(f"total_steps = {total_steps}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        accumulated_metrics = defaultdict(list)

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        max_advantage = 0
        min_advantage = 0

        for step, (completion, mask, reward, completion_log_probs, completion_ref_log_probs) in enumerate(progress_bar):
            total_step += 1
            completion_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors="pt")
            batch_tokens = completion_ids.to(device)
            prompt_ids = mask
            batch_advantages = torch.tensor(reward).to(device)
            max_advantage = max(max_advantage, torch.max(batch_advantages))
            min_advantage = min(min_advantage, torch.min(batch_advantages))
            avg_abs_advantage = torch.mean(torch.abs(batch_advantages))
            batch_log_probs = completion_log_probs[:,prompt_ids.shape[0]-1:].to(device)
            batch_ref_log_probs = completion_ref_log_probs[:,prompt_ids.shape[0]-1:].to(device)
            batch_masks = torch.ones_like(batch_ref_log_probs)
            
            outputs = model(batch_tokens, return_dict=True)
            logits = outputs.logits/temperature
            if tokenizer is not None and random.random() < 0.002:
                # print the first 10 and last 10 tokens of the batch
                print(f"first 10 tokens: {tokenizer.decode(batch_tokens[0,0:10])}")
                print(f"last 10 tokens: {tokenizer.decode(batch_tokens[0,-10:])}")

            logits = logits[:, prompt_ids.shape[0]-1:-1, :]
            #logits = filter_top_k_logits(logits, top_k)
            #logits = logits[:, prompt_ids.shape[1]:, :]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = torch.clamp(log_probs, min=log_prob_bias)
            target_tokens = completion_ids[:,prompt_ids.shape[0]:].to(device)
            log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1))
            log_probs = log_probs.squeeze(-1)

            old_probs = batch_log_probs

            per_token_logps = log_probs
            old_per_token_logps = old_probs

            ratio = torch.exp(per_token_logps - old_per_token_logps)
            
            # Calculate statistics on clipped values before applying clamp
            total_values = torch.sum(batch_masks).item()
            clipped_low = torch.sum((ratio < (1 - epsilon_low)) & batch_masks.bool()).item()
            clipped_high = torch.sum((ratio > (1 + epsilon_high)) & batch_masks.bool()).item()
            clipped_total = clipped_low + clipped_high
            
            clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
            # Calculate percentages
            pct_clipped_low = (clipped_low / total_values) * 100 if total_values > 0 else 0
            pct_clipped_high = (clipped_high / total_values) * 100 if total_values > 0 else 0
            pct_clipped_total = (clipped_total / total_values) * 100 if total_values > 0 else 0

            policy_loss_unclipped = ratio * batch_advantages.unsqueeze(-1)
            policy_loss_clipped = clipped_ratio * batch_advantages.unsqueeze(-1)
            policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped)
            mask_sum = batch_masks.sum()
            if random.random() < 0.02:
                try:
                    print(f"shape of completion_ids = {completion_ids.shape}, shape of clipped_ratio = {clipped_ratio.shape}, prompt_ids.shape = {prompt_ids.shape}")
                    print(f"shape of per_token_logps = {per_token_logps.shape}")
                    print(f"completion_log_probs = {old_per_token_logps[0,12:15]}\nref_log_probs = {batch_ref_log_probs[0,12:15]}\nper_token_logps = {per_token_logps[0,12:15]}")
                    print(f"device of completion_log_probs = {old_per_token_logps.device}, device of batch_ref_log_probs = {batch_ref_log_probs.device}, device of per_token_logps = {per_token_logps.device}")
                    print(f"mask_sum = {mask_sum}")
                    comp_ids = completion_ids[0,prompt_ids.shape[0]-1:prompt_ids.shape[0]+5]
                    str_comp_ids = tokenizer.decode(comp_ids)
                    print(f"str_comp_ids = {str_comp_ids}")
                    comp_ids_2 = completion_ids[0,prompt_ids.shape[0]-22:prompt_ids.shape[0]+25]
                    str_comp_ids_2 = tokenizer.decode(comp_ids_2)
                    print(f"str_comp_ids_2 = {str_comp_ids_2}")
                    # Calculate the norm of the differences between logprobs
                    logprob_diff = per_token_logps - old_per_token_logps
                    masked_logprob_diff = logprob_diff * batch_masks
                    logprob_diff_norm = torch.norm(masked_logprob_diff, p=2).item()
                    logprob_diff_mean = masked_logprob_diff.sum() / (mask_sum + 1e-8)
                    print(f"logprob_diff_norm = {logprob_diff_norm}")
                    print(f"logprob_diff_mean = {logprob_diff_mean.item()}")
                    print(f"pct_clipped_total = {pct_clipped_total}")
                    
                    # If we have reference model logprobs, also calculate difference with those
                    if beta > 0:
                        ref_logprob_diff = per_token_logps - batch_ref_log_probs
                        masked_ref_logprob_diff = ref_logprob_diff * batch_masks
                        ref_logprob_diff_norm = torch.norm(masked_ref_logprob_diff, p=2).item()
                        ref_logprob_diff_mean = masked_ref_logprob_diff.sum() / (mask_sum + 1e-8)
                        print(f"ref_logprob_diff_norm = {ref_logprob_diff_norm}")
                        print(f"ref_logprob_diff_mean = {ref_logprob_diff_mean.item()}")
                    
                    # Print clipping statistics
                    print(f"Clipped values: {pct_clipped_total:.2f}% (Low: {pct_clipped_low:.2f}%, High: {pct_clipped_high:.2f}%)")
                    print("--------------------------------")
                except:
                    pass

            if beta > 0:
                per_token_kl = (
                    torch.exp(batch_ref_log_probs - per_token_logps) - (batch_ref_log_probs - per_token_logps) - 1
                )
                policy_loss += beta * per_token_kl
                kl_loss = per_token_kl * batch_masks
                kl_loss_sum = kl_loss.sum() / mask_sum

            masked_policy_loss = policy_loss * batch_masks
            masked_policy_loss = masked_policy_loss * loss_multiplier

            loss = masked_policy_loss.sum() / (mask_sum + 1e-8)


            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item()
            accumulated_metrics["train/pct_clipped_total"].append(pct_clipped_total)
            accumulated_metrics["train/pct_clipped_low"].append(pct_clipped_low)
            accumulated_metrics["train/pct_clipped_high"].append(pct_clipped_high)
            accumulated_metrics["train/loss"].append(loss.item() * gradient_accumulation_steps)
            accumulated_metrics["train/avg_abs_advantage"].append(avg_abs_advantage.item())
            # Update on gradient accumulation steps
            if (total_step + 1) % gradient_accumulation_steps == 0 or total_step == total_steps - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                optimizer.zero_grad()

                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                accumulated_metrics["train/learning_rate"].append(current_lr)

                # Accumulate metrics instead of directly logging them
                accumulated_metrics["train/policy_loss"].append(masked_policy_loss.sum().item() / (mask_sum.item() + 1e-8))
                accumulated_metrics["train/grad_norm"].append(grad_norm.item())
                accumulated_metrics["train/mask_sum"].append(mask_sum.item())
                if beta > 0:
                    accumulated_metrics["train/kl_loss"].append(kl_loss_sum.item())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'policy_loss': masked_policy_loss.sum().item() / (mask_sum.item() + 1e-8),
                'clipped': f"{pct_clipped_total:.1f}%",
            })

        if use_wandb:
            # Calculate averages of accumulated metrics
            avg_metrics = {
                metric: sum(values) / len(values) if values else 0
                for metric, values in accumulated_metrics.items()
            }
            avg_metrics.update({
                "train/step": total_step,
                "train/epoch": epoch,
                "train/max_advantage": max_advantage,
                "train/min_advantage": min_advantage,
                "train/loss_std": torch.std(torch.tensor(accumulated_metrics["train/loss"])).item(),
            })
            wandb.log(avg_metrics, step=iteration)
            avg_epoch_loss = epoch_loss / len(dataloader)
            stats['epoch_losses'].append(avg_epoch_loss)
            if accumulated_metrics["train/policy_loss"]:
                avg_policy_loss = sum(accumulated_metrics["train/policy_loss"]) / len(accumulated_metrics["train/policy_loss"])
                stats['policy_losses'].append(avg_policy_loss)
            else:
                stats['policy_losses'].append(0.0)

    return stats

if __name__ == "__main__":
    import torch
    from outputs_dataset import create_dataloader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Sample data (similar to the example in outputs_dataset.py)
    output1 = torch.tensor([1, 2, 1, 2, 1, 2, 101, 102, 103, 102, 101])
    mask1 = torch.tensor([0, 1, 1, 1, 1, 1])
    
    output2 = torch.tensor([4, 5, 3, 4, 5, 3, 5, 6, 7, 202, 203, 204, 205])
    mask2 = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    outputs = [output1, output2]
    masks = [mask1, mask2]
    rewards = torch.tensor([1.0, 1.0])
    
    # Initialize wandb
    
    # Create dataloader
    batch_size = 2
    dataloader = create_dataloader(outputs, masks, rewards, batch_size, device)
    
    # Initialize a small model for testing
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Run training
    training_stats = train_with_grpo(
        model=model,
        dataloader=dataloader,
        num_epochs=2,
        optimizer=optimizer,
        gradient_accumulation_steps=1,
        epsilon_low=0.2,
        epsilon_high=0.2,
        max_grad_norm=1.0,
        device=device,
        use_wandb=False,
    )
    
    print("Training complete!")
    print(f"Final policy loss: {training_stats['policy_losses'][-1]}")
    wandb.finish()