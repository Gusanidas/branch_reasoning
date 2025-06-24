from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
from typing import List, Dict, Any, Optional, Union
from branch_reasoning.generation.completions import (
    PromptCompletion,
    ScoringData,
    Metadata,
    Branch,
    BranchedCompletion,
)
import torch
import json
import random
from torch.utils.data import Dataset
from branch_reasoning.prompts import system_prompt


class PromptCompletionDataset(Dataset):
    """Dataset that expands PromptCompletion objects into individual training examples."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        prompt_completions: List[PromptCompletion],
    ):
        print(f"In prompt_completion_dataset")
        self.tokenizer = tokenizer
        self.examples = []
        skip_count = 0

        for prompt_completion in prompt_completions:
            for branched_completion in prompt_completion.branched_completions:
                for branch in branched_completion.branches:
                    if branch.score is None or branch.log_probs is None:
                        skip_count += 1
                        continue

                    self.examples.append(
                        {
                            "prompt": prompt_completion.prompt,
                            "completion": branch.completion,
                            "log_probs": branch.log_probs,
                            "ref_log_probs": branch.ref_log_probs,
                            "score": branch.score,
                            "bare_prompt": prompt_completion.bare_prompt,
                        }
                    )
        print(f"Skipped {skip_count} examples")

    @classmethod
    def from_jsonl(
        cls, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, data_path: str
    ):
        """Create a PromptCompletionDataset from a .jsonl file."""
        prompt_completions = []

        with open(data_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_data = json.loads(line)
                    prompt_completion = deserialize_prompt_completion(json_data)
                    prompt_completions.append(prompt_completion)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")

        if not prompt_completions:
            raise ValueError(f"No valid JSON lines found in {data_path}")

        print(f"Loaded {len(prompt_completions)} valid entries from {data_path}")
        return cls(tokenizer, prompt_completions)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def deserialize_prompt_completion(data: Dict) -> PromptCompletion:
    """Converts a dictionary back to a PromptCompletion object."""

    scoring_data = ScoringData(
        nums=data["scoring_data"]["nums"], target=data["scoring_data"]["target"]
    )

    metadata = Metadata(
        solution=data["metadata"]["solution"], tag=data["metadata"]["tag"]
    )

    # Deserialize branched_completions
    branched_completions = []
    for bc in data["branched_completions"]:
        branches = []
        for b in bc["branches"]:
            branch = Branch(
                completion=b["completion"],
                log_probs=(
                    torch.tensor(b["log_probs"])
                    if b.get("log_probs") is not None
                    else None
                ),
                ref_log_probs=(
                    torch.tensor(b["ref_log_probs"])
                    if b.get("ref_log_probs") is not None
                    else None
                ),
                score=b.get("score"),
                key=b["key"],
                meta_scores=b.get("meta_scores", {}),
            )
            branches.append(branch)

        branched_completion = BranchedCompletion(
            branches=branches, score=bc.get("score")
        )
        branched_completions.append(branched_completion)

    return PromptCompletion(
        prompt=data["prompt"],
        scoring_data=scoring_data,
        metadata=metadata,
        branched_completions=branched_completions,
        bare_prompt=data.get("bare_prompt"),
        problem_id=data.get("problem_id"),
    )


def _format_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    completion: Optional[str] = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    add_generation_prompt = True
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
        add_generation_prompt = False
    formated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )
    assert isinstance(formated_prompt, str)
    return formated_prompt


def serialize_prompt_completion(pc: PromptCompletion) -> Dict[str, Any]:
    """Converts a PromptCompletion object to a JSON-serializable dictionary."""
    return {
        "prompt": pc.prompt,
        "scoring_data": {
            "nums": pc.scoring_data.nums,
            "target": pc.scoring_data.target,
        },
        "metadata": {"solution": pc.metadata.solution, "tag": pc.metadata.tag},
        "branched_completions": [
            {
                "branches": [
                    {
                        "completion": b.completion,
                        "log_probs": (
                            b.log_probs.tolist()
                            if hasattr(b, "log_probs") and b.log_probs is not None
                            else None
                        ),
                        "ref_log_probs": (
                            b.ref_log_probs.tolist()
                            if hasattr(b, "ref_log_probs")
                            and b.ref_log_probs is not None
                            else None
                        ),
                        "score": b.score,
                        "key": b.key,
                        "meta_scores": b.meta_scores,
                    }
                    for b in bc.branches
                ],
                "score": bc.score,
            }
            for bc in pc.branched_completions
        ],
        "bare_prompt": pc.bare_prompt,
        "problem_id": pc.problem_id,
    }


def collate_fn(
    batch,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    target_length: Optional[int] = None,
):
    """Collate function for batching training examples."""
    prompts = [example["prompt"] for example in batch]

    bare_prompts = [example.get("bare_prompt", None) for example in batch]
    if bare_prompts[0] is None:
        prompt_ids = tokenizer(
            [_format_prompt(prompt, tokenizer) for prompt in prompts],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        ).input_ids
        completions = [
            _format_prompt(example["prompt"], tokenizer, example["completion"])
            for example in batch
        ]
    else:
        prompt_ids = tokenizer(
            [_format_prompt(prompt, tokenizer) for prompt in prompts],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        ).input_ids
        completions = [
            _format_prompt(example["bare_prompt"], tokenizer, example["completion"])
            for example in batch
        ]

    tokenized_completions = tokenizer(
        completions,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        padding_side="right",
    )
    completion_ids = tokenized_completions.input_ids
    completion_masks = tokenized_completions.attention_mask[:, 1:]

    log_probs = [example["log_probs"] for example in batch]
    ref_log_probs = [example["ref_log_probs"] for example in batch]
    scores = [example["score"] for example in batch]
    if target_length is None:
        target_length = max([completion_ids[i].size(0) - 1 for i in range(len(batch))])
    else:
        # Pad completion_ids and completion_masks to target_length
        max_completion_length = target_length + 1

        # Pad or truncate completion_ids
        current_length = completion_ids.size(1)
        if current_length < max_completion_length:
            padding_size = max_completion_length - current_length
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            # Convert to int to ensure compatibility with torch.full
            pad_token_id = (
                int(pad_token_id) if isinstance(pad_token_id, (int, float)) else 0
            )
            padding = torch.full(
                (completion_ids.size(0), padding_size),
                pad_token_id,
                dtype=completion_ids.dtype,
                device=completion_ids.device,
            )
            completion_ids = torch.cat([completion_ids, padding], dim=1)
        elif current_length > max_completion_length:
            completion_ids = completion_ids[:, :max_completion_length]

        # Pad or truncate completion_masks
        current_mask_length = completion_masks.size(1)
        if current_mask_length < target_length:
            padding_size = target_length - current_mask_length
            padding = torch.zeros(
                (completion_masks.size(0), padding_size),
                dtype=completion_masks.dtype,
                device=completion_masks.device,
            )
            completion_masks = torch.cat([completion_masks, padding], dim=1)
        elif current_mask_length > target_length:
            completion_masks = completion_masks[:, :target_length]

    # Pad log_probs and ref_log_probs to target_length
    extended_log_probs = []
    extended_ref_log_probs = []

    for i in range(len(batch)):
        current_log_probs = log_probs[i]
        current_length = current_log_probs.size(0)

        if current_length < target_length:
            padding = torch.zeros(
                target_length - current_length,
                device=current_log_probs.device,
                dtype=current_log_probs.dtype,
            )
            extended_log_probs.append(torch.cat([current_log_probs, padding]))
        else:
            extended_log_probs.append(current_log_probs[:target_length])

        # Do the same for ref_log_probs if they exist
        if ref_log_probs[i] is not None:
            current_ref_log_probs = ref_log_probs[i]
            current_ref_length = current_ref_log_probs.size(0)

            if current_ref_length < target_length:
                padding = torch.zeros(
                    target_length - current_ref_length,
                    device=current_ref_log_probs.device,
                    dtype=current_ref_log_probs.dtype,
                )
                extended_ref_log_probs.append(
                    torch.cat([current_ref_log_probs, padding])
                )
            else:
                extended_ref_log_probs.append(current_ref_log_probs[:target_length])
        else:
            extended_ref_log_probs.append(None)

        completion_masks[i, : prompt_ids[i].size(0) - 2] = 0

    scores = torch.tensor(scores)
    extended_log_probs = torch.stack(extended_log_probs)
    extended_ref_log_probs = (
        torch.stack(extended_ref_log_probs)
        if extended_ref_log_probs[0] is not None
        else None
    )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "completion_masks": completion_masks,
        "log_probs": extended_log_probs,
        "ref_log_probs": extended_ref_log_probs,
        "scores": scores,
    }


class JsonlDataset(Dataset):
    """A dataset for loading a .jsonl file and converting to PromptCompletion objects."""

    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_data = json.loads(line)
                    prompt_completion = deserialize_prompt_completion(json_data)
                    self.data.append(prompt_completion)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")

        if not self.data:
            raise ValueError(f"No valid JSON lines found in {data_path}")

        print(f"Loaded {len(self.data)} valid entries from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
