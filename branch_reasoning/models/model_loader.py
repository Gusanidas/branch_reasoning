from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


def get_models_and_tokenizers(model_name, reference_model_name, tokenizer_name, use_bfloat16, beta, use_vllm=False):
    """
    Load models and tokenizers for either HuggingFace or vLLM usage.
    
    Args:
        model_name: Name or path of the model to load
        reference_model_name: Name or path of the reference model (or None to use model_name)
        tokenizer_name: Name or path of the tokenizer (or None to use model_name)
        use_bfloat16: Whether to use bfloat16 precision
        beta: Beta value for reference model comparison (if > 0, load reference model)
        use_vllm: Whether the models will be used with vLLM
        
    Returns:
        Tuple of (model, reference_model, tokenizer)
        For vLLM, model and reference_model will be None, only tokenizer is returned
    """
    if use_bfloat16:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load reference model if needed
    reference_model = None
    if beta > 0:
        if reference_model_name is None:
            reference_model_name = model_name
        if use_bfloat16:
            reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name, torch_dtype=torch.bfloat16)
        else:
            reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name)
        reference_model.eval()  # Set to evaluation mode
        for param in reference_model.parameters():
            param.requires_grad = False
    
    # Load tokenizer
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For vLLM, we don't need to return the models, just the tokenizer
    if use_vllm:
        # Save the model and tokenizer to the checkpoints directory if using vLLM
        os.makedirs("./checkpoints", exist_ok=True)
        model.save_pretrained("./checkpoints")
        tokenizer.save_pretrained("./checkpoints")

    return model, reference_model, tokenizer