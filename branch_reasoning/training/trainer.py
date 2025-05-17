import torch
import random
import time
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import wandb
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR

from branch_reasoning.generation.completions import PromptCompletion, Branch
from branch_reasoning.prompts import system_prompt

class PromptCompletionDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_completions: List[PromptCompletion]):
        self.tokenizer = tokenizer
        self.examples = []
        
        for prompt_completion in prompt_completions:
            for branched_completion in prompt_completion.branched_completions:
                for branch in branched_completion.branches:
                    if branch.score is None or branch.log_probs is None:
                        continue
                    
                    
                    self.examples.append({
                        "prompt": prompt_completion.prompt,
                        "completion": branch.completion,
                        "log_probs": branch.log_probs,
                        "ref_log_probs": branch.ref_log_probs,
                        "score": branch.score
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]



def _fromat_prompt(prompt: str, tokenizer: PreTrainedTokenizer, completion: Optional[str] = None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    add_generation_prompt = True
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
        add_generation_prompt = False
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=True
    )

def collate_fn(batch, tokenizer: PreTrainedTokenizer):
    prompts = [example["prompt"] for example in batch]
    for i, prompt in enumerate(prompts[:-1]):
        if prompts[i+1] != prompt:
            raise ValueError(f"Prompt {prompt} is not the same as prompt {prompts[i+1]}")

    bare_prompts = [example.get("bare_prompt", None) for example in batch]
    if bare_prompts[0] is None:
        prompt_ids = tokenizer(
            [_fromat_prompt(prompt, tokenizer) for prompt in prompts],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right"
        ).input_ids
        completions = [_fromat_prompt(example["prompt"], tokenizer, example["completion"]) for example in batch]
    else:
        prompt_ids = tokenizer(
            [_fromat_prompt(prompt, tokenizer) for prompt in prompts],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right"
        ).input_ids
        completions = [_fromat_prompt(example["bare_prompt"], tokenizer, example["completion"]) for example in batch]
    
    tokenized_completions = tokenizer(completions, add_special_tokens=False, return_tensors="pt", padding=True, padding_side="right")
    completion_ids = tokenized_completions.input_ids
    completion_masks = tokenized_completions.attention_mask

    log_probs = [example["log_probs"] for example in batch]
    ref_log_probs = [example["ref_log_probs"] for example in batch]
    scores = [example["score"] for example in batch]
    target_length = max([completion_ids[i].size(0) - prompt_ids[i].size(0) for i in range(len(batch))])
    
    
    extended_log_probs = []
    extended_ref_log_probs = []
    
    for i in range(len(batch)):
        current_log_probs = log_probs[i]
        current_length = current_log_probs.size(0)
        
        if current_length < target_length:
            padding = torch.zeros(target_length - current_length, device=current_log_probs.device, 
                                 dtype=current_log_probs.dtype)
            extended_log_probs.append(torch.cat([current_log_probs, padding]))
        else:
            extended_log_probs.append(current_log_probs[:target_length])
        
        # Do the same for ref_log_probs if they exist
        if ref_log_probs[i] is not None:
            current_ref_log_probs = ref_log_probs[i]
            current_ref_length = current_ref_log_probs.size(0)
            
            if current_ref_length < target_length:
                padding = torch.zeros(target_length - current_ref_length, device=current_ref_log_probs.device,
                                     dtype=current_ref_log_probs.dtype)
                extended_ref_log_probs.append(torch.cat([current_ref_log_probs, padding]))
            else:
                extended_ref_log_probs.append(current_ref_log_probs[:target_length])
        else:
            extended_ref_log_probs.append(None)

    scores = torch.tensor(scores)
    extended_log_probs = torch.stack(extended_log_probs)
    extended_ref_log_probs = torch.stack(extended_ref_log_probs) if extended_ref_log_probs[0] is not None else None
    
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "completion_masks": completion_masks,
        "log_probs": extended_log_probs,
        "ref_log_probs": extended_ref_log_probs,
        "scores": scores
    }


def _make_dataloader(
    tokenizer: PreTrainedTokenizer,
    prompt_completions: List[PromptCompletion],
    batch_size: int,
) -> DataLoader:
    """
    Create a DataLoader for training with GRPO algorithm.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt_completions: List of PromptCompletion objects
        batch_size: Batch size for training
        
    Returns:
        DataLoader object containing batches of data for training
    """
    dataset = PromptCompletionDataset(tokenizer, prompt_completions)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        # We don't shuffle the data here because we want to group prompts together
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    
    return dataloader

def create_multiple_dataloaders(
    tokenizer: PreTrainedTokenizer,
    prompt_completions: List[PromptCompletion],
    batch_size: int,
) -> List[DataLoader]:
    dataloaders = []
    for prompt_completion in prompt_completions:
        # Create a simple dataset for each list
        dataloaders.append(_make_dataloader(tokenizer, [prompt_completion], batch_size))
    
    # Create a wrapper to iterate through all dataloaders
    class CombinedLoader:
        def __init__(self, loaders):
            self.loaders = loaders
        
        def __iter__(self):
            for loader in self.loaders:
                for batch in loader:
                    yield batch
        
        def __len__(self):
            return sum(len(loader) for loader in self.loaders)
    
    return CombinedLoader(dataloaders)

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
    batch_size = training_args.get("batch_size", 1)
    # Initialize statistics
    stats = {
        'epoch_losses': [],
        'policy_losses': []
    }

    completions_per_prompt = len(prompt_completions[0].branched_completions)

    # Group prompt_completions for batching
    total_prompts = len(prompt_completions) * num_epochs
    t0 = time.time()
    print(f"----------0----------")
    print(f"Inside train_with_grpo, total_prompts = {total_prompts}, temperature = {temperature}")
    prompt_steps = 0
    total_steps = 0
    skipped_branches = 0
    updates = 0
    max_tokens = 0
    max_abs_loss = 0
    for epoch in range(num_epochs):
        if training_args.get("full_last_epoch", False) and epoch == num_epochs-1:
            gradient_accumulation_steps = len(prompt_completions)
        epoch_loss = 0
        optimizer.zero_grad()
        
        accumulated_metrics = defaultdict(list)
        
        max_advantage = 0
        min_advantage = 0
        #combined_dataloader = create_multiple_dataloaders(tokenizer, prompt_completions, batch_size)
        dataloader_list = [_make_dataloader(tokenizer, [prompt_completion], batch_size) for prompt_completion in prompt_completions]
        
        
        for dataloader in dataloader_list:
            prompt_steps += 1
            for batch in dataloader:
                total_steps += 1

                masks = batch["completion_masks"].to(device)
                prompt_ids = batch["prompt_ids"].to(device)
                completion_ids = batch["completion_ids"].to(device)
                completion_masks = batch["completion_masks"].to(device)

                #print(f"Shape of prompt_ids = {prompt_ids.shape}")
                #print(f"First 5 prompt_ids = {prompt_ids[:,:5]}")
                #print(f"Last 5 prompt_ids = {prompt_ids[:,-5:]}")
                #print(f"Shape of completion_ids = {completion_ids.shape}")
                #print(f"First 5 completion_ids = {completion_ids[:,:5]}")
                #print(f"Last 5 completion_ids = {completion_ids[:,-5:]}")
                #print(f"Shape of completion_masks = {completion_masks.shape}")
                #print(f"First 5 masks = {completion_masks[:,:5]}")
                #print(f"Last 5 masks = {completion_masks[:,-5:]}")

                #print(f"Type of batch['scores'] = {type(batch['scores'])}")
                #print(f"Type of batch['scores'][0] = {type(batch['scores'][0])}")
                #print(f"Type of batch['scores'][0].item() = {type(batch['scores'][0].item())}")
                advantage = batch["scores"].to(device)

                # Get the advantage/score
                max_advantage = max(max_advantage, advantage.max().item())
                min_advantage = min(min_advantage, advantage.min().item())
                avg_abs_advantage = torch.abs(advantage).mean().item()

                # Get the stored log probabilities
                log_probs = batch["log_probs"].to(device)
                ref_log_probs = batch["ref_log_probs"].to(device) if batch["ref_log_probs"] is not None else None

                #print(f"Shape of log_probs = {log_probs.shape}")
                #print(f"Shape of ref_log_probs = {ref_log_probs.shape}")


                # Get model outputs
                full_ids = completion_ids
                max_tokens = max(max_tokens, full_ids.shape[1])
                outputs = model(full_ids, return_dict=True)#, attention_mask=masks)
                logits = outputs.logits / temperature

                # Slice logits to match the completion part only
                logits = logits[:, prompt_ids.shape[1]-1:-1, :]
                #print(f"Shape of logits after slicing = {logits.shape}")
                # Calculate log probabilities
                model_log_probs = F.log_softmax(logits, dim=-1)
                #print(f"Shape of model_log_probs = {model_log_probs.shape}")
                #model_log_probs = model_log_probs[0]

                # Gather the log probabilities for the actual tokens
                target_tokens = completion_ids[:,prompt_ids.shape[1]:]
                target_tokens_detokenized = tokenizer.batch_decode(target_tokens, skip_special_tokens=False)
                model_log_probs = model_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1))
                #print(f"Shape of model_log_probs after gathering = {model_log_probs.shape}")
                model_log_probs = model_log_probs.squeeze(-1)
                #print(f"Shape of model_log_probs after squeezing = {model_log_probs.shape}")
                # Calculate ratio between new and old policy
                ratio = torch.exp(model_log_probs - log_probs)
                #print(f"Shape of ratio = {ratio.shape}")
                masks = completion_masks[:,prompt_ids.shape[1]:]
                #print(f"Shape of masks = {masks.shape}")

                # Calculate statistics on clipped values
                total_values = torch.sum(masks).item()
                if random.random() < 0.002 or total_steps <2:
                    print(f"First target tokens detokenized: {target_tokens_detokenized[:10]}")
                    print(f"Shape of masks = {masks.shape}")
                    print(f"Shape of ratio = {ratio.shape}")
                    print(f"Shape of log_probs = {log_probs.shape}")
                    print(f"Shape of model_log_probs = {model_log_probs.shape}")
                    print(f"Shape of completion_ids = {completion_ids.shape}")
                    print(f"Shape of prompt_ids = {prompt_ids.shape}")
                    print(f"first 5 log_probs: {log_probs[:,:10]}")
                    print(f"first 5 model_log_probs: {model_log_probs[:,:10]}")
                    if ref_log_probs is not None:
                        print(f"first 5 ref_log_probs: {ref_log_probs[:,:10]}")

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
                policy_loss_unclipped = ratio * advantage.unsqueeze(-1)
                policy_loss_clipped = clipped_ratio * advantage.unsqueeze(-1)
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
                max_abs_loss = max(max_abs_loss, torch.abs(loss).item() * gradient_accumulation_steps * completions_per_prompt)

                # Log stats
                accumulated_metrics["train/pct_clipped_total"].append(pct_clipped_total)
                accumulated_metrics["train/pct_clipped_low"].append(pct_clipped_low)
                accumulated_metrics["train/pct_clipped_high"].append(pct_clipped_high)
                accumulated_metrics["train/loss"].append(loss.item() * gradient_accumulation_steps * completions_per_prompt)
                accumulated_metrics["train/avg_abs_advantage"].append(avg_abs_advantage)
            
            if prompt_steps % gradient_accumulation_steps == gradient_accumulation_steps-1 or prompt_steps == total_prompts-1:
                updates += 1
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
                "train/updates": updates,
                "train/epoch": epoch,
                "train/loss_std": torch.std(torch.tensor(accumulated_metrics["train/loss"])).item() if accumulated_metrics["train/loss"] else 0,
                "train/max_abs_loss": max_abs_loss,
                "train/max_tokens": max_tokens,
                "train/max_advantage": max_advantage,
                "train/min_advantage": min_advantage,
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

    