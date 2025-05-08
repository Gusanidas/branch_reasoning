from typing import Optional, Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from branch_reasoning.generation.completions import Branch, PromptCompletion



def get_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str = "cuda",
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Gets log probabilities for a given text using the provided model and tokenizer.
    
    Args:
        model: The language model to use
        tokenizer: The tokenizer associated with the model
        text: The text to analyze for log probabilities 
        device: The device to run inference on ("cuda", "cpu", etc.)
        temperature: Temperature parameter to control the distribution of probabilities.
                    Higher values produce more uniform distributions, lower values make
                    the distribution more peaked. Default is 1.0 (no change).
        
    Returns:
        A tensor containing log probabilities for each token in the text
    """
    if device is not None:
        model = model.to(device)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Get model outputs without sampling
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        
    # Get logits and shift to align with targets
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    
    # Apply temperature to scale logits before calculating log probabilities
    # Higher temperature makes distribution more uniform, lower makes it more peaked
    scaled_logits = shifted_logits / temperature
    
    # Calculate log probabilities
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    
    # Get log probabilities of actual tokens
    token_log_probs = torch.gather(
        log_probs, 
        dim=2, 
        index=shifted_targets.unsqueeze(-1)
    ).squeeze(-1)
    
    return token_log_probs[0]  # Return the tensor without batch dimension


def populate_branch_log_probs(
    branch: Branch,
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reference_model: Optional[PreTrainedModel] = None,
    device: str = "cuda",
    temperature: float = 1.0
) -> Branch:
    """
    Populates the log_probs and ref_log_probs fields of a Branch object.
    
    Args:
        branch: The Branch object to populate
        model: The main language model
        tokenizer: The tokenizer associated with the models
        reference_model: Optional reference model for calculating reference log probs
        device: The device to run inference on ("cuda", "cpu", etc.)
        temperature: Temperature parameter to control the distribution of probabilities.
                    Higher values produce more uniform distributions, lower values make
                    the distribution more peaked. Default is 1.0 (no change).
        
    Returns:
        The updated Branch object with populated log_probs and ref_log_probs
    """
    # Calculate log probabilities for the main model
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    log_probs = get_log_probs(
        model=model,
        tokenizer=tokenizer,
        text=branch.completion,
        device=device,
        temperature=temperature
    )
    branch.log_probs = log_probs[len(prompt_tokens)-1:]
    # Calculate reference log probabilities if a reference model is provided
    if reference_model is not None:
        branch.ref_log_probs = get_log_probs(
            model=reference_model,
            tokenizer=tokenizer,
            text=branch.completion,
            device=device,
            temperature=temperature
        )
        branch.ref_log_probs = branch.ref_log_probs[len(prompt_tokens)-1:]

def populate_log_probs(
    prompt_completions: List[PromptCompletion],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reference_model: Optional[PreTrainedModel] = None,
    device: str = "cuda",
    temperature: float = 1.0
) -> List[PromptCompletion]:
    for prompt_completion in prompt_completions:
        for branched_completion in prompt_completion.branched_completions:
            for branch in branched_completion.branches:
                populate_branch_log_probs(
                    branch=branch,
                    prompt=prompt_completion.prompt,
                    model=model,
                    tokenizer=tokenizer,
                    reference_model=reference_model,
                    device=device,
                    temperature=temperature
                )
