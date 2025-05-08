from collections import defaultdict
from typing import List, Tuple
from branch_reasoning.generation.completions import PromptCompletion, BranchedCompletion
import math
import random

def _populate_avg_scores(
    prompt_completion_list: List[PromptCompletion],
    **kwargs,
):
    for prompt_completion in prompt_completion_list:
        for branched_completion in prompt_completion.branched_completions:
            avg_score = 0
            for branch in branched_completion.branches:
                avg_score += branch.score
            avg_score /= len(branched_completion.branches)
            branched_completion.score = avg_score

def _normalize_scores(
    prompt_completion_list: List[PromptCompletion],
    normalize_by_prompt: bool = True, #TODO: have this in the main variables
    **kwargs,
):
    if normalize_by_prompt:
        for prompt_completion in prompt_completion_list:
            scores = []
            for branched_completion in prompt_completion.branched_completions:
                scores.append(branched_completion.score)
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
            for branched_completion in prompt_completion.branched_completions:
                branched_completion.score = (branched_completion.score - mean_score) / std_score if std_score != 0 else 0
    else:
        scores = []
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                scores.append(branched_completion.score)
        mean_score = sum(scores) / len(scores)
        std_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                branched_completion.score = (branched_completion.score - mean_score) / std_score if std_score != 0 else 0

def _divide_branched_scores(
    prompt_completion_list: List[PromptCompletion],
    **kwargs,
):
    for prompt_completion in prompt_completion_list:
        for branched_completion in prompt_completion.branched_completions:
            total = 0
            for branch in branched_completion.branches:
                total += branch.score 
            ratio = total/branched_completion.score if branched_completion.score != 0 else 1
            for branch in branched_completion.branches:
                branch.score /= ratio * len(branched_completion.branches) if ratio != 0 else 1

def _initialize_scores(
    prompt_completion_list: List[PromptCompletion],
    init_value: float = 0,
    **kwargs,
):
    for prompt_completion in prompt_completion_list:
        for branched_completion in prompt_completion.branched_completions:
            for branch in branched_completion.branches:
                branch.score = init_value

def _gather_metrics(
    prompt_completion_list: List[PromptCompletion],
    wandb_logs: dict,
    mark: str = "",
    **kwargs,
):
    once = True
    tmax_score = -1000
    tmin_score = 1000
    for prompt_completion in prompt_completion_list:
        scores = []
        b_scores = []
        for branched_completion in prompt_completion.branched_completions:
            scores.append(branched_completion.score)
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        tmax_score = max(tmax_score, max_score)
        tmin_score = min(tmin_score, min_score)
        std_score = math.sqrt(sum((score - avg_score) ** 2 for score in scores) / len(scores))
        if once:
            print(f"Once scores: {scores}")
            print(f"Once avg_score: {avg_score}, max_score: {max_score}, min_score: {min_score}, std_score: {std_score}")
            once = False
        wandb_logs[f"post_score_{mark}/avg_score"] += avg_score
        wandb_logs[f"post_score_{mark}/max_score"] += max_score
        wandb_logs[f"post_score_{mark}/min_score"] += min_score
        wandb_logs[f"post_score_{mark}/std_score"] += std_score
        wandb_logs[f"post_score_{mark}/avg_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/max_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/min_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/std_score_steps"] += 1
    print(f"tmax_score: {tmax_score}, tmin_score: {tmin_score}, mark: {mark}")
    print("----")


class CompletionScorer:
    def __init__(self, scoring_functions, normalize_by_prompt: bool = True):
        """
        Initialize the CompletionScorer with tokenizer and scoring functions.

        Args:
            scoring_functions: A list of scoring functions that each accept
                              (completion, wandb_logs, **kwargs) and return a score
        """
        self.scoring_functions = scoring_functions
        self.normalize_by_prompt = normalize_by_prompt #TODO: have this in the main variables

    def score_completions(
        self, prompt_completion_list: List[PromptCompletion],
        wandb_logs: dict,
        **kwargs,
    ):
        _initialize_scores(prompt_completion_list, init_value=0)
        for prompt_completion in prompt_completion_list:
            tag = prompt_completion.metadata.tag
            prompt = prompt_completion.prompt
            for branched_completion in prompt_completion.branched_completions:
                for scoring_function in self.scoring_functions:
                    function_score = scoring_function(
                        branched_completion=branched_completion,
                        prompt=prompt,
                        wandb_logs=wandb_logs,
                        scoring_data=prompt_completion.scoring_data,
                        **kwargs,
                    )

                    function_name = scoring_function.__name__
                    if random.random() < 0.001:
                        print(f" PRint inside : <> function_name: {function_name}, function_score: {function_score}") #TODO: Remove
                    wandb_logs[f"score/{function_name}"] += function_score
                    wandb_logs[f"score/total_score"] += function_score
                    wandb_logs[f"score/total_score_steps"] += 1
                    wandb_logs[f"score/{function_name}_steps"] += 1
                    if tag is not None:
                        wandb_logs[f"{tag}_score/{function_name}"] += function_score
                        wandb_logs[f"{tag}_score/{function_name}_steps"] += 1
                        wandb_logs[f"{tag}_score/total_score"] += function_score
                        wandb_logs[f"{tag}_score/total_steps"] += 1

        _populate_avg_scores(prompt_completion_list)
        _gather_metrics(prompt_completion_list, wandb_logs, mark="pre")
        _normalize_scores(prompt_completion_list, self.normalize_by_prompt)
        _divide_branched_scores(prompt_completion_list)
        _gather_metrics(prompt_completion_list, wandb_logs)
        return prompt_completion_list

