import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from typing import Dict, List, Optional
import argparse
from pathlib import Path
from datasets import load_dataset
from functools import partial
from itertools import cycle

# Import evaluation modules
from branch_reasoning.scoring import (
    CompletionScorer,
    match_format_exactly,
    rate_countdown_answer,
    match_format_approximately,
    match_format_loosely,
    score_branch_format_approx,
    score_branch_format_loose,
    score_branch_format,
)
from branch_reasoning.countdown_task import transform_countdown_data
from branch_reasoning.prompts import (
    base_prompt,
    get_format_and_examples,
)
from branch_reasoning.generation.completions import generate_completions


def clean_prompt_from_examples(prompt: str) -> str:
    """Remove text between 'Example:' and 'Task:' from the prompt."""
    example_start = prompt.find("Example:")
    task_start = prompt.find("Task:")
    
    if example_start != -1 and task_start != -1 and example_start < task_start:
        # Remove the text between "Example:" and "Task:"
        return prompt[:example_start] + prompt[task_start:]
    
    return prompt

class MaskedCompletionDataset(Dataset):
    """Dataset that masks prompt tokens and only trains on completion tokens."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
        remove_examples: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load data from JSONL file
        with open(data_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                self.examples.append(
                    {"prompt": data[prompt_key], "completion": data[completion_key]}
                )
                if remove_examples:
                    self.examples.append(
                        {"prompt": clean_prompt_from_examples(data[prompt_key]), "completion": data[completion_key]}
                    )

        random.shuffle(self.examples)
        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        prompt = example["prompt"]
        completion = example["completion"]

        # Combine prompt and completion
        full_text = prompt + completion

        # Tokenize the full text
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (clone input_ids)
        labels = encodings["input_ids"].clone()

        # Tokenize just the prompt to find where to mask
        prompt_encodings = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt"
        )

        prompt_length = prompt_encodings["input_ids"].shape[1]

        # Mask the prompt tokens in labels (-100 is ignored by loss function)
        labels[0, :prompt_length] = -100

        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class CustomDataCollator:
    """Custom data collator that preserves our label masking."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Stack all tensors
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([f[key] for f in features])
        return batch


class EvaluationCallback(TrainerCallback):
    """Callback to evaluate at specific epochs."""

    def __init__(self, eval_func, eval_epochs=[1, 3]):
        self.eval_func = eval_func
        self.eval_epochs = eval_epochs
        self.last_epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch > self.last_epoch and current_epoch in self.eval_epochs:
            self.last_epoch = current_epoch
            print(f"\n=== Evaluating after epoch {current_epoch} ===")
            self.eval_func()


def evaluate_model(
    model,
    tokenizer,
    eval_data,
    completion_scorer,
    branching_factor,
    max_branching_points,
    max_tokens=1536,
    total_completions=32,
    completions_per_prompt=4,
    batch_size=32,
    temperature=0.7,
):
    """Evaluate the model on countdown tasks."""
    print("\nGenerating completions for evaluation...")

    # Generate completions
    prompt_completions = generate_completions(
        dataset=eval_data,
        model=model,
        tokenizer=tokenizer,
        max_len=max_tokens,
        gen_batch_size= batch_size,
        total_completions=total_completions,  # Generate 4 completions per prompt
        completions_per_prompt=completions_per_prompt,
        current_iter=0,
        branching_factor=branching_factor,
        max_branching_points=max_branching_points,
        temperature=temperature,
        branch_completions=True,
        wandb_logging=False,
    )
    #for pc in prompt_completions:
    #    print()
    #    print()
    #    print("+"*100)
    #    print("+"*100)
    #    print(f"Prompt: \n{pc.prompt}")
    #    for i, bc in enumerate(pc.branched_completions):
    #        print()
    #        print("="*100)
    #        print(f"Branched completion no {i}: \n")
    #        for j, b in enumerate(bc.branches):
    #            print()
    #            print("-"*100)
    #            print(f"Branch key: {b.key}")
    #            print(f"Branch no {j}: \n{b.completion}")
    #            print()
    #            print("-"*100)
    #    print("+"*100)
    #    print("+"*100)
    #    print()
    #    print()

    # Score completions
    print("Scoring completions...")
    prompt_completions = completion_scorer.score_completions(
        prompt_completions, normalize_by_prompt=True,
    )

    # Calculate and print average scores
    total_score = 0
    total_count = 0
    score_breakdown = {}

    for pc in prompt_completions:
        for bc in pc.branched_completions:
            total_score += bc.score
            total_count += 1

            # Track individual scoring function results
            for branch in bc.branches:
                for key, value in branch.meta_scores.items():
                    if key not in score_breakdown:
                        score_breakdown[key] = {"total": 0, "count": 0}
                    score_breakdown[key]["total"] += value
                    score_breakdown[key]["count"] += 1

    avg_score = total_score / total_count if total_count > 0 else 0
    print(f"\nAverage completion score: {avg_score:.4f}")
    print(f"Total completions evaluated: {total_count}")

    print("\nScore breakdown by function:")
    for func_name, stats in score_breakdown.items():
        avg = stats["total"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {func_name}: {avg:.4f}")

    return avg_score


def main():
    # Define variables directly instead of using argument parsing
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    #model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    #model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path = "../merged_branches.jsonl"
    output_dir = "./finetuned_model"
    reference_model_name = "Gusanidas/branch-grpo-model-qwen-3b-branch"
    num_epochs = 2
    batch_size = 1
    learning_rate = 4e-5
    warmup_steps = 40
    save_steps = 500
    logging_steps = 100
    max_length = 1536
    gradient_accumulation_steps = 22
    use_bfloat16 = True  # Use bfloat16 precision for training
    max_tokens = max_length
    total_completions = 32
    gen_batch_size = 32
    temperature = 0.7
    completions_per_prompt = 8
    # Evaluation parameters
    branching_factor = 2
    max_branching_points = 3
    num_eval_prompts = 8  # Number of prompts to use for evaluation

    # Initialize tokenizer and model
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate dtype
    if use_bfloat16:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        print("Model loaded with bfloat16 precision")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model loaded with default precision")

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print(f"Loading dataset from {data_path}")
    dataset = MaskedCompletionDataset(
        data_path=data_path, tokenizer=tokenizer, max_length=max_length
    )

    # Split dataset into train and eval (90/10 split)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Create data collator
    data_collator = CustomDataCollator(tokenizer)

    # Load evaluation dataset (countdown tasks)
    print("Loading evaluation dataset...")
    HF_DATASET_REPO_ID = "Gusanidas/countdown-tasks-dataset-med-vl6"
    eval_dataset = load_dataset(HF_DATASET_REPO_ID)["train"]

    # Prepare evaluation prompts
    format_prompt, examples = get_format_and_examples(
        branch_completions=True,
        max_branching_points=max_branching_points,
        branching_factor=branching_factor,
    )
    examples = ["Example:\n" + example for example in examples]
    examples = examples + [""] * len(examples)

    eval_data = transform_countdown_data(
        eval_dataset,
        base_prompt_template=base_prompt,
        template_func=None,
        format_prompt=format_prompt,
        examples=examples,
    )
    #eval_data = eval_data[:num_eval_prompts]
    eval_data = cycle(eval_data)

    # Set up completion scorer
    pscore_branch_format_approx = partial(
        score_branch_format_approx,
        max_branches=max_branching_points,
        branch_factor=branching_factor,
    )
    pscore_branch_format_loose = partial(
        score_branch_format_loose,
        max_branches=max_branching_points,
        branch_factor=branching_factor,
    )
    pscore_branch_format = partial(
        score_branch_format,
        max_branches=max_branching_points,
        branch_factor=branching_factor,
    )

    pscore_branch_format_approx.__name__ = "score_branch_format_approx"
    pscore_branch_format_loose.__name__ = "score_branch_format_loose"
    pscore_branch_format.__name__ = "score_branch_format"

    completion_scorer = CompletionScorer(
        scoring_functions=[
            match_format_exactly,
            rate_countdown_answer,
            match_format_approximately,
            match_format_loosely,
            pscore_branch_format_approx,
            pscore_branch_format_loose,
            pscore_branch_format,
        ]
    )

    # Evaluate before training
    print("\n=== Evaluating BEFORE training ===")
    eval_func = lambda: evaluate_model(
        model,
        tokenizer,
        eval_data,
        completion_scorer,
        branching_factor,
        max_branching_points,
        max_tokens=max_tokens,
        total_completions=total_completions,
        batch_size=gen_batch_size,
        temperature=temperature,
        completions_per_prompt=completions_per_prompt,
    )
    initial_score = eval_func()

    # Set up training arguments (without saving)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=999999,  # Effectively disable saving
        save_total_limit=0,  # Don't save any checkpoints
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
    )

    # Create evaluation callback
    eval_callback = EvaluationCallback(eval_func, eval_epochs=[1, 3])

    # Create trainer with callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        data_collator=data_collator,
        #callbacks=[eval_callback],
    )

    # Train the model
    print("\nStarting training...")
    trainer.train()

    # Final evaluation after all epochs
    print("\n=== Evaluating AFTER training (3 epochs) ===")
    final_score = eval_func()

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Initial score (before training): {initial_score:.4f}")
    print(f"Final score (after 3 epochs): {final_score:.4f}")
    print(f"Improvement: {final_score - initial_score:.4f}")
    print("=" * 50)

    # Save model to Hugging Face
    print(f"\nSaving model to Hugging Face as: {reference_model_name}")
    try:
        model.push_to_hub(reference_model_name)
        tokenizer.push_to_hub(reference_model_name)
        print(f"✅ Model and tokenizer successfully saved to: {reference_model_name}")
    except Exception as e:
        print(f"❌ Error saving to Hugging Face: {e}")
        print("Make sure you're logged in with `huggingface-cli login`")

    print("\nTraining complete! Model saved to Hugging Face.")


if __name__ == "__main__":
    main()

