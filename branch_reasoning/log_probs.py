from typing import Optional, Union, List, Dict, Tuple, cast
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from branch_reasoning.generation.completions import Branch, PromptCompletion
from branch_reasoning.prompts import system_prompt


def get_log_probs_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: Union[str, List[str]],
    device: str = "cuda",
    temperature: float = 1.0,
    max_length: Optional[int] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Gets log probabilities for one or more texts using the provided model and tokenizer.

    Args:
        model: The language model to use
        tokenizer: The tokenizer associated with the model
        texts: A single text string or list of texts to analyze for log probabilities
        device: The device to run inference on ("cuda", "cpu", etc.)
        temperature: Temperature parameter to control the distribution of probabilities.
                    Higher values produce more uniform distributions, lower values make
                    the distribution more peaked. Default is 1.0 (no change).
        max_length: Maximum sequence length for padding. If None, uses the longest sequence.

    Returns:
        If single text: A tensor containing log probabilities for each token
        If multiple texts: A list of tensors, one per text, containing log probabilities
    """
    # Handle single text input
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    if device is not None:
        model = model.to(device)

    # Tokenize all texts at once with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        padding_side="right",
        max_length=max_length,
        truncation=True if max_length else False,
    )

    input_ids = cast(torch.Tensor, inputs["input_ids"])
    attention_mask = cast(torch.Tensor, inputs["attention_mask"])
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    # Get model outputs without sampling
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get logits and shift to align with targets
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    shifted_attention_mask = attention_mask[:, 1:]

    # Apply temperature to scale logits before calculating log probabilities
    scaled_logits = shifted_logits / temperature

    # Calculate log probabilities
    log_probs = F.log_softmax(scaled_logits, dim=-1)

    # Get log probabilities of actual tokens
    token_log_probs = torch.gather(
        log_probs, dim=2, index=shifted_targets.unsqueeze(-1)
    ).squeeze(-1)

    # Apply attention mask to ignore padding tokens
    token_log_probs = token_log_probs * shifted_attention_mask

    # Split results back into individual sequences
    results = []
    for i in range(len(texts)):
        # Get the actual length of this sequence (excluding padding)
        seq_len = shifted_attention_mask[i].sum().item()
        # Extract only the non-padded tokens
        seq_log_probs = token_log_probs[i, :seq_len]
        results.append(seq_log_probs)

    # Return single tensor if single input, otherwise list
    return results[0] if single_input else results


def get_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str = "cuda",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Gets log probabilities for a given text using the provided model and tokenizer.
    (Backward compatible single-text version)

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
    result = get_log_probs_batch(model, tokenizer, text, device, temperature)
    return cast(
        torch.Tensor, result
    )  # Safe cast since single text returns single tensor


def populate_branches_log_probs_batch(
    branches_data: List[Tuple[Branch, str]],  # List of (branch, prompt) tuples
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reference_model: Optional[PreTrainedModel] = None,
    device: str = "cuda",
    temperature: float = 1.0,
    batch_size: int = 8,
) -> None:
    """
    Populates log_probs and ref_log_probs fields for multiple Branch objects in batches.

    Args:
        branches_data: List of tuples containing (Branch object, prompt string)
        model: The main language model
        tokenizer: The tokenizer associated with the models
        reference_model: Optional reference model for calculating reference log probs
        device: The device to run inference on ("cuda", "cpu", etc.)
        temperature: Temperature parameter for probability distribution
        batch_size: Number of branches to process in each batch
    """
    # Process branches in batches
    for i in range(0, len(branches_data), batch_size):
        batch = branches_data[i : i + batch_size]
        texts = []
        prompt_lens = []

        # Prepare texts for this batch
        for branch, prompt in batch:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_tokens = cast(
                torch.Tensor,
                tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ],
            )
            prompt_lens.append(prompt_tokens.size(1))

            messages.append({"role": "assistant", "content": branch.completion})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            texts.append(text)

        # Get log probabilities for all texts in batch
        log_probs_list = get_log_probs_batch(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            temperature=temperature,
        )

        # Assign log probs to branches
        for j, (branch, _) in enumerate(batch):
            branch.log_probs = log_probs_list[j].cpu()

        # Calculate reference log probabilities if reference model provided
        if reference_model is not None:
            ref_log_probs_list = get_log_probs_batch(
                model=reference_model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                temperature=temperature,
            )

            for j, (branch, _) in enumerate(batch):
                branch.ref_log_probs = ref_log_probs_list[j].cpu()


def populate_branch_log_probs(
    branch: Branch,
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reference_model: Optional[PreTrainedModel] = None,
    device: str = "cuda",
    temperature: float = 1.0,
) -> Branch:
    """
    Populates the log_probs and ref_log_probs fields of a Branch object.
    (Backward compatible single-branch version)

    Args:
        branch: The Branch object to populate
        prompt: The prompt text
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
    populate_branches_log_probs_batch(
        [(branch, prompt)],
        model=model,
        tokenizer=tokenizer,
        reference_model=reference_model,
        device=device,
        temperature=temperature,
        batch_size=1,
    )
    return branch


def populate_log_probs(
    prompt_completions: List[PromptCompletion],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    reference_model: Optional[PreTrainedModel] = None,
    batch_size: int = 8,
    device: str = "cuda",
    temperature: float = 1.0,
) -> List[PromptCompletion]:
    """
    Populates log probabilities for a list of PromptCompletion objects using batching.

    Args:
        prompt_completions: List of PromptCompletion objects to process
        model: The main language model
        tokenizer: The tokenizer associated with the models
        reference_model: Optional reference model for calculating reference log probs
        batch_size: Number of branches to process in each batch
        device: The device to run inference on ("cuda", "cpu", etc.)
        temperature: Temperature parameter for probability distribution

    Returns:
        The updated list of PromptCompletion objects with populated log probabilities
    """
    model = model.eval()
    if reference_model is not None:
        reference_model = reference_model.eval()

    # Collect all branches with their prompts
    branches_data = []
    for prompt_completion in prompt_completions:
        prompt = (
            prompt_completion.bare_prompt
            if prompt_completion.bare_prompt is not None
            else prompt_completion.prompt
        )
        for branched_completion in prompt_completion.branched_completions:
            for branch in branched_completion.branches:
                branches_data.append((branch, prompt))

    # Process all branches in batches
    populate_branches_log_probs_batch(
        branches_data,
        model=model,
        tokenizer=tokenizer,
        reference_model=reference_model,
        device=device,
        temperature=temperature,
        batch_size=batch_size,
    )

    return prompt_completions
