from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def get_models_and_tokenizers(model_name, reference_model_name, tokenizer_name, use_bfloat16, beta):
    if use_bfloat16:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
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
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, reference_model, tokenizer