from typing import List, Dict, Any, Union, Optional, Tuple, NamedTuple, Iterator
import html
import wandb
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from collections import defaultdict
from dataclasses import dataclass
import itertools

from branch_reasoning.generation.text_generation import hf_generate_text
from branch_reasoning.generation.branching import QueueDataset, get_new_branches


@dataclass
class Branch:
    completion: str
    log_probs: Optional[torch.Tensor]
    ref_log_probs: Optional[torch.Tensor]
    score: Optional[float]


@dataclass
class BranchedCompletion:
    branches: List[Branch]
    score: Optional[float]


class ScoringData(NamedTuple):
    nums: List[float]
    target: float


class Metadata(NamedTuple):
    solution: str
    tag: str


@dataclass
class PromptCompletion:
    prompt: str
    scoring_data: ScoringData
    metadata: Metadata
    branched_completions: List[BranchedCompletion]


def _calculate_batch_parameters(
    completions_per_prompt: int, gen_batch_size: int, total_completions: int
) -> Tuple[int, int, int, int]:
    if completions_per_prompt > gen_batch_size:
        prompt_repetitions = completions_per_prompt // gen_batch_size
        prompts_per_batch = 1
    else:
        prompt_repetitions = 1
        prompts_per_batch = gen_batch_size // completions_per_prompt

    generation_iter = total_completions // (gen_batch_size * prompt_repetitions)
    num_completions = min(completions_per_prompt, gen_batch_size)

    return prompt_repetitions, prompts_per_batch, generation_iter, num_completions


def _fetch_batch(
    dataset: Iterator[Dict[str, Any]], prompts_per_batch: int
) -> Tuple[List[str], List[List[float]], List[float], List[str], List[str]]:
    prompts = []
    numbers = []
    targets = []
    tags = []
    solutions = []
    for _ in range(prompts_per_batch):
        item = next(dataset)
        prompts.append(item["question"])
        numbers.append(item["numbers"])
        targets.append(item["target"])
        tags.append(item["tag"])
        solutions.append(item["solution"])
    return prompts, numbers, targets, tags, solutions


def _perform_branching(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    all_completions: Dict[str, str],
    max_branching_points: int,
    branching_factor: int,
    gen_batch_size: int,
    max_len: int,
    device: Optional[str] = None,
    generation_args: Dict[str, Any] = {},
) -> Dict[str, str]:
    new_branches = get_new_branches(
        all_completions,
        branching_factor=branching_factor,
        max_branching_points=max_branching_points,
    )
    queue_dataset = QueueDataset(
        tokenizer=tokenizer,
        batch_size=gen_batch_size,
        # If the branching point is at the end of the sequence, we skip branching it.
        max_seq_len=max_len - 5,
    )
    queue_dataset.add_sequences(new_branches.items())
    while not queue_dataset.is_empty():
        keys, batch = queue_dataset.next_batch()

        completions = hf_generate_text(
            model,
            tokenizer,
            batch,
            num_completions=1,
            max_seq_len=max_len,
            device=device,
            generation_args=generation_args,
        )

        new_completions = {}
        for key, completion in zip(keys, completions):
            new_completions[key] = completion

        all_completions.update(new_completions)
        new_branches = get_new_branches(
            new_completions,
            branching_factor=branching_factor,
            max_branching_points=max_branching_points,
        )
        queue_dataset.add_sequences(new_branches.items())
    return all_completions


def _log_statistics(
    all_completions: Dict[str, str],
    original_no_branches: int,
    current_iter: int,
    branch_ratio: float,
    example_completion_html: wandb.Html,
    example_completion_text: wandb.Html,
) -> None:
    branch_ratio = len(all_completions) / original_no_branches
    keys, completions = zip(*all_completions.items())
    total_completion_length = sum(len(c) for c in completions)
    total_completions_count = len(completions)
    avg_completion_length = (
        total_completion_length / total_completions_count
        if total_completions_count > 0
        else 0
    )
    wandb.log(
        {
            "avg_completion_length": avg_completion_length,
            "branch_ratio": branch_ratio,
            "total_completions": total_completions_count,
            "example_completion_html": example_completion_html,
            "example_completion_text": example_completion_text,
        },
        step=current_iter,
    )


def _pack_into_return_datatypes(
    all_completions: Dict[str, str],
    all_prompts: Dict[str, str],
    all_numbers: Dict[str, List[float]],
    all_targets: Dict[str, float],
    all_tags: Dict[str, str],
    all_solutions: Dict[str, str],
) -> List[PromptCompletion]:
    branched_completions = dict()
    for k, v in all_completions.items():
        base_key, completion_key, *_ = k.split("_")
        gen_key = base_key + "_" + completion_key
        branched_completions[gen_key] = branched_completions.get(
            gen_key, BranchedCompletion(branches=[], score=None)
        )
        branch = Branch(completion=v, log_probs=None, ref_log_probs=None, score=None)
        branched_completions[gen_key].branches.append(branch)

    prompt_completions = dict()
    for k, v in branched_completions.items():
        base_key, completion_key, *_ = k.split("_")
        gen_key = base_key + "_" + completion_key
        prompt = all_prompts[base_key]
        numbers = all_numbers[base_key]
        target = all_targets[base_key]
        solution = all_solutions[base_key]
        tag = all_tags[base_key]
        prompt_completions[base_key] = prompt_completions.get(
            base_key,
            PromptCompletion(
                prompt=prompt,
                scoring_data=ScoringData(nums=numbers, target=target),
                metadata=Metadata(solution=solution, tag=tag),
                branched_completions=[],
            ),
        )
        prompt_completions[base_key].branched_completions.append(v)

    return list(prompt_completions.values())


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: itertools.cycle,
    total_completions: int,
    completions_per_prompt: int,
    gen_batch_size: int,
    current_iter: int,
    max_len: int,
    device: Optional[str] = None,
    wandb_logging: bool = True,
    branch_completions: bool = True,
    branching_factor: int = 2,
    max_branching_points: int = 3,
    generation_args: dict = {},
) -> List[PromptCompletion]:
    """
    Generate completions for the given dataset.
    """

    (
        prompt_repetitions,
        prompts_per_batch,
        generation_iter,
        num_completions,
    ) = _calculate_batch_parameters(
        completions_per_prompt, gen_batch_size, total_completions
    )

    all_completions = {}
    all_prompts = {}
    all_numbers = {}
    all_targets = {}
    all_tags = {}
    all_solutions = {}

    total_keys = 0

    for i in range(generation_iter):
        prompts, numbers, targets, tags, solutions = _fetch_batch(
            dataset, prompts_per_batch
        )
        for j in range(prompt_repetitions):
            completions = hf_generate_text(
                model,
                tokenizer,
                prompts,
                num_completions=num_completions,
                max_seq_len=max_len,
                device=device,
                generation_args=generation_args,
            )

            if wandb_logging and j == 0:  # TODO: Remove one of the two.
                decoded_completion = tokenizer.decode(
                    completions[0], skip_special_tokens=True
                )
                example_completion_html = wandb.Html(
                    f"<pre>{html.escape(decoded_completion)}</pre>"
                )
                example_completion_text = wandb.Html(f"```\n{decoded_completion}\n```")

            for k in range(prompts_per_batch):
                total_keys += 1
                base_key = f"{total_keys}#{current_iter}"
                all_prompts[base_key] = prompts[k]
                all_numbers[base_key] = numbers[k]
                all_targets[base_key] = targets[k]
                all_tags[base_key] = tags[k]
                all_solutions[base_key] = solutions[k]
                for kk in range(num_completions):
                    completion_key = f"{base_key}_{kk}_" + "_".join(
                        ["0"] * max_branching_points
                    )
                    all_completions[completion_key] = completions[
                        k * num_completions + kk
                    ]

    original_no_branches = len(all_completions)
    if branch_completions:
        all_completions = _perform_branching(
            model,
            tokenizer,
            all_completions,
            max_branching_points,
            branching_factor,
            gen_batch_size,
            max_len,
            device,
            generation_args,
        )

    if wandb_logging:
        _log_statistics(
            all_completions,
            original_no_branches,
            current_iter,
            branch_ratio,
            example_completion_html,
            example_completion_text,
        )

    # Pack into the return datatypes
    return _pack_into_return_datatypes(
        all_completions,
        all_prompts,
        all_numbers,
        all_targets,
        all_tags,
        all_solutions,
    )


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class DummyDatasetCycler:
        def __iter__(self) -> Iterator[Dict[str, Any]]:
            return self

        def __next__(self) -> Dict[str, Any]:
            return {
                "question": "What is 2 + 2?",
                "numbers": [2, 2],
                "target": 4,
                "tag": "math",
                "solution": "4",
            }

    dataset = DummyDatasetCycler()

    total_completions = 4
    completions_per_prompt = 2
    gen_batch_size = 2
    current_iter = 0
    temperature = 0.7
    top_p = 1.0
    top_k = 50
    max_length = 32
    device = "cpu"
    wandb_logging = False
    branch_completions = False
    branching_factor = 2
    max_branching_points = 1

    print("Generating completions...")
    results = generate_completions(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        total_completions=total_completions,
        completions_per_prompt=completions_per_prompt,
        gen_batch_size=gen_batch_size,
        current_iter=current_iter,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_length=max_length,
        device=device,
        wandb_logging=wandb_logging,
        branch_completions=branch_completions,
        branching_factor=branching_factor,
        max_branching_points=max_branching_points,
    )
