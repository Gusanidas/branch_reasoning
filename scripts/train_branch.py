import time
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
from collections import defaultdict
from datasets import concatenate_datasets, load_dataset
from itertools import cycle

from branch_reasoning.utils.countdown_task import transform_countdown_data, apply_r1_template
from branch_reasoning.utils.prompts import (
    base_prompt,
    single_branch_examples,
    multi_branch_examples,
    single_branch_format_prompt,
    multi_branch_format_prompt,
)
from branch_reasoning.utils.utils import linear_warmup_decay
from branch_reasoning.models.model_loader import get_models_and_tokenizers
from branch_reasoning.generation.completions import generate_completions

model_name = "gpt2"
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
#model_name = "Qwen/Qwen2.5-0.5B"
#model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "Qwen/Qwen2.5-3B-Instruct"
#model_name = "Gusanidas/branch-grpo-model-qwen-3b-br"
#model_name = "Gusanidas/branch-grpo-model-qwen-3b"
#model_name = "Qwen/Qwen2.5-Math-1.5B"
# model_name = "google/gemma-3-4b-it"  # Change this to test different models
#model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
#model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# Available models:
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "google/gemma-3-1b-it"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name = "Gusanidas/branch-grpo-model-qwen-3b"#-pre"
#tokenizer_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
#tokenizer_name = "Qwen/Qwen2.5-1.5B-Instruct"
#tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_name = "Qwen/Qwen2.5-3B-Instruct"
#tokenizer_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

reference_model_name = None

device = "mps"

save_interval = 20
hf_repo_name = "Gusanidas/branch-grpo-model-qwen-3b-branch"
hf_dataset_name = "Gusanidas/countdown-tasks-dataset-balanced"

iterations = 500

# Generation parameters
# In one iteration, how many solutions to generate. Several branches can belong to the same completion.
total_completions = 2
# How many completions to generate in one batch.
gen_batch_size = 2
# How many completions to generate per prompt.
no_completions = 2
# How many branches to generate for each branching point.
branching_factor = 2
# Maximum number of branching points.
max_branching_points = 3
# Maximum length of the completion.
max_len = 768
# Generation arguments.
opt_generation_args = {
    "temperature": 1.0,
    "top_p": 1,
    "top_k": 50,
}



wandb_logging = False
reuse_completions = False
branch_completions = True
use_bfloat16 = True

mixing_factor = 0.9
drop_rate = 0.4

max_gradient_accumulation_steps = 8
gradient_accumulation_steps = 8
train_batch_size = 1
num_epochs = 2
learning_rate = 3e-6
epsilon_low = 0.2
epsilon_high = 0.2
beta = 0.01
max_grad_norm = 5.0
divide_by_std_by_prompt = True
subtract_mean_by_prompt = True
weight_decay = 0.01
betas = (0.9, 0.99)


if no_completions % gen_batch_size != 0 and gen_batch_size % no_completions != 0:
    raise ValueError(
        "no_completions must be divisible by gen_batch_size or gen_batch_size must be divisible by no_completions"
    )

if no_completions > gen_batch_size:
    prompt_repetitions = no_completions // gen_batch_size
    prompts_per_batch = 1
else:
    prompt_repetitions = 1
    prompts_per_batch = gen_batch_size // no_completions

print(f"prompts_per_batch: {prompts_per_batch}")
print(f"prompt_repetitions: {prompt_repetitions}")

if total_completions % (prompts_per_batch * prompt_repetitions) != 0:
    raise ValueError(
        "total_completions must be divisible by prompts_per_batch * prompt_repetitions"
    )

generation_iter = total_completions // (gen_batch_size * prompt_repetitions)
print(f"generation_iter: {generation_iter}")

model, reference_model, tokenizer = get_models_and_tokenizers(model_name, reference_model_name, tokenizer_name, use_bfloat16, beta)
# Initialize optimizer
#optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
## SGD
##optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
#scheduler = linear_warmup_decay(optimizer, 0.05 * iterations, iterations)

t0 = time.time()
t00 = time.time()


task_dataset = load_dataset(hf_dataset_name)["train"]
examples = ["Example:\n" + example for example in multi_branch_examples] + [""]*5
dataset = transform_countdown_data(task_dataset, base_prompt_template=base_prompt, template_func=apply_r1_template, format_prompt=multi_branch_format_prompt, examples=examples)
cycling_dataset = cycle(dataset)

if wandb_logging:
    wandb.init(
        project="branch_grpo_vast_branch",
        name="run_921",
        config={
            "model_name": model_name,
            "iterations": iterations,
            "total_completions": total_completions,
            "gen_batch_size": gen_batch_size,
            "no_completions": no_completions,
            "train_batch_size": train_batch_size,
            "prompts_per_batch": prompts_per_batch,
            "prompt_repetitions": prompt_repetitions,
            "generation_iter": generation_iter,
            "learning_rate": learning_rate,
        },
    )

t0 = time.time()
t2, t3 = t0, t0

for i in range(iterations):
    prompt_completions = generate_completions(
        model,
        tokenizer,
        cycling_dataset,
        total_completions,
        completions_per_prompt = no_completions,
        gen_batch_size = gen_batch_size,
        current_iter = i,
        max_len = max_len,
        generation_args = opt_generation_args,
        device = device,
        wandb_logging = wandb_logging,
        branch_completions = branch_completions,
        branching_factor = branching_factor,
        max_branching_points = max_branching_points,
    )

    print(f"Len of prompt_completions: {len(prompt_completions)}")
    for prompt_completion in prompt_completions:
        print("-" * 100)
        print(f"PROMPT")
        print(prompt_completion.prompt)
        print(prompt_completion.metadata.solution)
        print(f"len of branched_completions: {len(prompt_completion.branched_completions)}")
        branched_completions = prompt_completion.branched_completions
        for branched_completion in branched_completions:
            print("-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            print(f"BRANCHED COMPLETION")
            print(f"len of branches: {len(branched_completion.branches)}")
            branches = branched_completion.branches
            for branch in branches:
                print("-" * 100)
                print(f"BRANCH")
                print(branch.completion)
                print(f"score: {branch.score}")
                print(f"log_probs: {branch.log_probs}")
                print(f"ref_log_probs: {branch.ref_log_probs}")
                print("-" * 100)
            print("-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print("^^" * 100)