from collections import defaultdict
from typing import List, Tuple
from branch_reasoning.generation.completions import PromptCompletion, BranchedCompletion, Branch, ScoringData, Metadata
import math
import random
from enum import Enum
from branch_reasoning.config import BranchedCompletionScoringMethod, NormalizeMethod
# Enum with the different ways to populate the branched completion scores

def _populate_brached_completion_scores(
    prompt_completion_list: List[PromptCompletion],
    scoring_method: BranchedCompletionScoringMethod = BranchedCompletionScoringMethod.AVERAGE,
    **kwargs,
):
    if scoring_method == BranchedCompletionScoringMethod.AVERAGE:
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                avg_score = 0
                for branch in branched_completion.branches:
                    avg_score += branch.score
                avg_score /= len(branched_completion.branches)
                branched_completion.score = avg_score
    elif scoring_method == BranchedCompletionScoringMethod.MAX:
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                max_score = max(branched_completion.branches, key=lambda x: x.score).score
                branched_completion.score = max_score


def _normalize_scores_by_prompt(
    prompt_completion_list: List[PromptCompletion],
    normalize_all_branches: bool = False,
    normalize_prompt_ratio: float = 1,
    **kwargs,
):
    """Normalize scores by individual prompt."""
    if normalize_all_branches:
        # Normalize by prompt, considering all branches
        for prompt_completion in prompt_completion_list:
            scores = []
            for branched_completion in prompt_completion.branched_completions:
                for branch in branched_completion.branches:
                    scores.append(branch.score)
            
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
            
            for branched_completion in prompt_completion.branched_completions:
                for branch in branched_completion.branches:
                    prev_score = branch.score
                    new_score = (branch.score - mean_score) / std_score if std_score != 0 else 0.0
                    branch.score = new_score * normalize_prompt_ratio + prev_score * (1 - normalize_prompt_ratio)
    else:
        # Normalize by prompt, considering only branched_completions
        for prompt_completion in prompt_completion_list:
            scores = []
            for branched_completion in prompt_completion.branched_completions:
                scores.append(branched_completion.score)
            
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
            
            for branched_completion in prompt_completion.branched_completions:
                prev_score = branched_completion.score
                new_score = (branched_completion.score - mean_score) / std_score if std_score != 0 else 0.0
                branched_completion.score = new_score * normalize_prompt_ratio + prev_score * (1 - normalize_prompt_ratio)


def _normalize_scores_by_all_prompts(
    prompt_completion_list: List[PromptCompletion],
    normalize_all_branches: bool = False,
    **kwargs,
):
    """Normalize scores across all prompts."""
    if normalize_all_branches:
        # Normalize across all prompts, considering all branches
        scores = []
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                for branch in branched_completion.branches:
                    scores.append(branch.score)
        
        mean_score = sum(scores) / len(scores)
        std_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
        
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                for branch in branched_completion.branches:
                    branch.score = (branch.score - mean_score) / std_score if std_score != 0 else 0.0
    else:
        # Normalize across all prompts, considering only branched_completions
        scores = []
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                scores.append(branched_completion.score)
        
        mean_score = sum(scores) / len(scores)
        std_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
        
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                branched_completion.score = (branched_completion.score - mean_score) / std_score if std_score != 0 else 0.0


def _normalize_scores(
    prompt_completion_list: List[PromptCompletion],
    normalize_by_prompt: NormalizeMethod = NormalizeMethod.PROMPT,
    normalize_prompt_ratio: float = 1,
    normalize_all_branches: bool = False,
    **kwargs,
):
    """
    Normalize scores by individual prompt, across all prompts, or a combination of both.
    
    Args:
        prompt_completion_list: List of PromptCompletion objects
        normalize_by_prompt: Enum specifying normalization method (PROMPT, ALL, or BOTH)
        normalize_prompt_ratio: Float between 0 and 1 for blending when using BOTH method
        normalize_all_branches: If True, consider all branches. If False, consider only branched_completions.
        **kwargs: Additional keyword arguments passed to the underlying functions.
    """
    if normalize_by_prompt != NormalizeMethod.BOTH:
        normalize_prompt_ratio = 1
    if normalize_by_prompt == NormalizeMethod.ALL or normalize_by_prompt == NormalizeMethod.BOTH:
        _normalize_scores_by_all_prompts(prompt_completion_list, normalize_all_branches, **kwargs)
    if normalize_by_prompt == NormalizeMethod.PROMPT or normalize_by_prompt == NormalizeMethod.BOTH:
        _normalize_scores_by_prompt(prompt_completion_list, normalize_all_branches, normalize_prompt_ratio, **kwargs)

def _divide_branched_scores(
    prompt_completion_list: List[PromptCompletion],
    scoring_method: BranchedCompletionScoringMethod = BranchedCompletionScoringMethod.AVERAGE,
    normalize_all_branches: bool = False,
    **kwargs,
):
    if normalize_all_branches:
        return
    if scoring_method == BranchedCompletionScoringMethod.AVERAGE:
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                total = 0
                for branch in branched_completion.branches:
                    total += branch.score 
                if branched_completion.score != 0:  
                    ratio = total/branched_completion.score
                    for branch in branched_completion.branches:
                        branch.score /= ratio * len(branched_completion.branches) if ratio != 0 else 1.0
                else:
                    ratio = 1
                    for branch in branched_completion.branches:
                        branch.score = 0.0
    elif scoring_method == BranchedCompletionScoringMethod.MAX:
        for prompt_completion in prompt_completion_list:
            for branched_completion in prompt_completion.branched_completions:
                max_score = max(branched_completion.branches, key=lambda x: x.score).score
                branches = []
                for branch in branched_completion.branches:
                    if branch.score == max_score:
                        branch.score = branched_completion.score
                        branches.append(branch)
                        break
                branched_completion.branches = branches

def _initialize_scores(
    prompt_completion_list: List[PromptCompletion],
    init_value: float = 0,
    **kwargs,
):
    for prompt_completion in prompt_completion_list:
        for branched_completion in prompt_completion.branched_completions:
            branched_completion.score = init_value
            for branch in branched_completion.branches:
                branch.score = init_value
                branch.meta_scores = defaultdict(float)

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
            new_b_scores = []
            for branch in branched_completion.branches:
                new_b_scores.append(branch.score)
            t_avg_score = sum(new_b_scores) / len(new_b_scores)
            t_std = math.sqrt(sum((score - t_avg_score) ** 2 for score in new_b_scores) / len(new_b_scores))
            b_scores += new_b_scores
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        tmax_score = max(tmax_score, max_score)
        tmin_score = min(tmin_score, min_score)
        std_score = math.sqrt(sum((score - avg_score) ** 2 for score in scores) / len(scores))
        if once:
            print(f"Once scores: {scores}")
            print(f"Once avg_score: {avg_score}, max_score: {max_score}, min_score: {min_score}, std_score: {std_score}")
            print(f"Once b_scores: {b_scores}")
            once = False
        wandb_logs[f"post_score_{mark}/avg_score"] += avg_score
        wandb_logs[f"post_score_{mark}/max_score"] += max_score
        wandb_logs[f"post_score_{mark}/min_score"] += min_score
        wandb_logs[f"post_score_{mark}/std_score"] += std_score
        wandb_logs[f"post_score_{mark}/stds_zero"] += 1 if std_score == 0 else 0
        wandb_logs[f"post_score_{mark}/stds_small"] += 1 if std_score < 0.04 else 0
        wandb_logs[f"post_score_{mark}/avg_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/max_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/min_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/std_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/stds_zero_steps"] += 1
        wandb_logs[f"post_score_{mark}/stds_small_steps"] += 1


        avg_b_score = sum(b_scores) / len(b_scores)
        max_b_score = max(b_scores)
        min_b_score = min(b_scores)
        std_b_score = math.sqrt(sum((score - avg_b_score) ** 2 for score in b_scores) / len(b_scores))
        wandb_logs[f"post_score_{mark}/avg_b_score"] += avg_b_score
        wandb_logs[f"post_score_{mark}/max_b_score"] += max_b_score
        wandb_logs[f"post_score_{mark}/min_b_score"] += min_b_score
        wandb_logs[f"post_score_{mark}/std_b_score"] += std_b_score
        wandb_logs[f"post_score_{mark}/avg_b_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/max_b_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/min_b_score_steps"] += 1
        wandb_logs[f"post_score_{mark}/std_b_score_steps"] += 1

    wandb_logs[f"post_score_{mark}/tmax_score"] += tmax_score
    wandb_logs[f"post_score_{mark}/tmin_score"] += tmin_score
    wandb_logs[f"post_score_{mark}/tmax_score_steps"] += 1
    wandb_logs[f"post_score_{mark}/tmin_score_steps"] += 1
    print(f"tmax_score: {tmax_score}, tmin_score: {tmin_score}, mark: {mark}")
    print("----")

def _gather_branch_metrics(
    prompt_completion_list: List[PromptCompletion],
    wandb_logs: dict,
    **kwargs,
):
    base_scores = []
    branched_scores = []
    for prompt_completion in prompt_completion_list:
        for branched_completion in prompt_completion.branched_completions:
            for branch in branched_completion.branches:
                if branch.key:
                    k = branch.key
                    base_key, completion_key, *branch_keys = k.split("_")
                    if not all([branch_key.isdigit() for branch_key in branch_keys]):
                        continue
                    if all([int(branch_key) < 1 for branch_key in branch_keys]):
                        base_scores.append(branch.score)
                    else:
                        branched_scores.append(branch.score)
    avg_base_score = sum(base_scores) / len(base_scores) if base_scores else 0
    avg_branched_score = sum(branched_scores) / len(branched_scores) if branched_scores else 0
    wandb_logs[f"branch_scores/avg_base_score"] += avg_base_score
    wandb_logs[f"branch_scores/avg_branched_score"] += avg_branched_score

class CompletionScorer:
    def __init__(self, scoring_functions, scoring_variables: dict = {}):
        """
        Initialize the CompletionScorer with tokenizer and scoring functions.

        Args:
            scoring_functions: A list of scoring functions that each accept
                              (completion, wandb_logs, **kwargs) and return a score
        """
        self.scoring_functions = scoring_functions
        self.normalize_by_prompt = scoring_variables.get("normalize_by_prompt", NormalizeMethod.PROMPT)
        self.normalize_prompt_ratio = scoring_variables.get("normalize_prompt_ratio", 0.5)
        self.normalize_all_branches = scoring_variables.get("normalize_all_branches", False)
        self.scoring_method = scoring_variables.get("scoring_method", BranchedCompletionScoringMethod.AVERAGE)

    def score_completions(
        self, prompt_completion_list: List[PromptCompletion],
        wandb_logs: dict = defaultdict(float),
        **kwargs,
    ) -> List[PromptCompletion]:
        _initialize_scores(prompt_completion_list, init_value=0)
        for prompt_completion in prompt_completion_list:
            tag = prompt_completion.metadata.tag
            prompt = prompt_completion.prompt
            solved_by_at_least_one = False
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
                    wandb_logs[f"score/{function_name}"] += function_score
                    wandb_logs[f"score/total_score"] += function_score
                    wandb_logs[f"score/total_score_steps"] += 1
                    wandb_logs[f"score/{function_name}_steps"] += 1
                    wandb_logs[f"score/score_calls_count"] += 1
                    if tag is not None:
                        wandb_logs[f"{tag}_score/{function_name}"] += function_score
                        wandb_logs[f"{tag}_score/{function_name}_steps"] += 1
                        wandb_logs[f"{tag}_score/total_score"] += function_score
                        wandb_logs[f"{tag}_score/total_steps"] += 1

                    if function_name == "rate_countdown_answer":
                        solved_by_at_least_one = solved_by_at_least_one or function_score > 0
            wandb_logs[f"score/solved_by_at_least_one"] += solved_by_at_least_one
            wandb_logs[f"score/solved_by_at_least_one_steps"] += 1
        _populate_brached_completion_scores(prompt_completion_list, self.scoring_method)
        _gather_metrics(prompt_completion_list, wandb_logs, mark="pre")
        _gather_branch_metrics(prompt_completion_list, wandb_logs)
        _normalize_scores(prompt_completion_list, self.normalize_by_prompt, self.normalize_prompt_ratio, self.normalize_all_branches)
        _divide_branched_scores(prompt_completion_list, self.scoring_method, self.normalize_all_branches)
        _gather_metrics(prompt_completion_list, wandb_logs, mark="post")
        return prompt_completion_list

if __name__ == "__main__":
    # Test the _normalize_scores function
    # Create sample data for testing
    def create_test_data():
        prompt_completions = []
        
        # Create first prompt completion
        metadata1 = Metadata(tag="test1", solution="4")
        prompt1 = "What is 2+2?"
        
        # Create branches with different scores
        branch1_1 = Branch(completion="4", score=0.8, key="test1_4", ref_log_probs = None, log_probs = None, meta_scores = {})
        branch1_2 = Branch(completion="Four", score=0.6, key="test1_four", ref_log_probs = None, log_probs = None, meta_scores = {})
        branch1_3 = Branch(completion="2+2=4", score=0.9, key="test1_2+2=4", ref_log_probs = None, log_probs = None, meta_scores = {})
        
        branch2_1 = Branch(completion="The answer is 4", score=0.7, key="test2_the_answer_is_4", ref_log_probs = None, log_probs = None, meta_scores = {})
        branch2_2 = Branch(completion="It equals 4", score=0.5, key="test2_it_equals_4", ref_log_probs = None, log_probs = None, meta_scores = {})
        
        # Create branched completions
        branched_comp1 = BranchedCompletion(branches=[branch1_1, branch1_2, branch1_3], score=0.8)
        branched_comp2 = BranchedCompletion(branches=[branch2_1, branch2_2], score=0.7)
        
        prompt_comp1 = PromptCompletion(
            prompt=prompt1,
            branched_completions=[branched_comp1, branched_comp2],
            metadata=metadata1,
            scoring_data=ScoringData(
                nums=[2, 2],
                target=4
            )
        )
        
        # Create second prompt completion
        metadata2 = Metadata(tag="test2", solution="6")
        prompt2 = "What is 3+3?"
        
        branch3_1 = Branch(completion="6", score=1.2, key="test2_6", ref_log_probs = None, log_probs = None, meta_scores = {})
        branch3_2 = Branch(completion="Six", score=1.0, key="test2_six", ref_log_probs = None, log_probs = None, meta_scores = {})
        
        branch4_1 = Branch(completion="The answer is 6", score=1.5, key="test2_the_answer_is_6", ref_log_probs = None, log_probs = None, meta_scores = {}   )
        branch4_2 = Branch(completion="It equals 6", score=0.3, key="test2_it_equals_6", ref_log_probs = None, log_probs = None, meta_scores = {})
        branch4_3 = Branch(completion="3+3=6", score=1.8, key="test2_3+3=6", ref_log_probs = None, log_probs = None, meta_scores = {})
        
        branched_comp3 = BranchedCompletion(branches=[branch3_1, branch3_2], score=1.2)
        branched_comp4 = BranchedCompletion(branches=[branch4_1, branch4_2, branch4_3], score=1.5)
        
        prompt_comp2 = PromptCompletion(
            prompt=prompt2,
            branched_completions=[branched_comp3, branched_comp4],
            metadata=metadata2,
            scoring_data=ScoringData(
                nums=[3, 3],
                target=6
            )
        )
        
        prompt_completions = [prompt_comp1, prompt_comp2]
        return prompt_completions
    
    def calculate_stats(prompt_completion_list, level="branch"):
        """Calculate mean and std for branch scores or branched completion scores"""
        scores = []
        
        if level == "branch":
            for prompt_completion in prompt_completion_list:
                for branched_completion in prompt_completion.branched_completions:
                    for branch in branched_completion.branches:
                        scores.append(branch.score)
        else:  # branched completion level
            for prompt_completion in prompt_completion_list:
                for branched_completion in prompt_completion.branched_completions:
                    scores.append(branched_completion.score)
        
        if not scores:
            return 0, 0
            
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = math.sqrt(variance)
        
        return mean, std
    
    def print_all_scores(prompt_completion_list, title):
        """Print all scores for debugging"""
        print(f"\n{title}")
        print("=" * 50)
        
        for i, prompt_completion in enumerate(prompt_completion_list):
            print(f"Prompt {i+1}: {prompt_completion.prompt}")
            for j, branched_completion in enumerate(prompt_completion.branched_completions):
                print(f"  Branched Completion {j+1} (score: {branched_completion.score:.4f}):")
                for k, branch in enumerate(branched_completion.branches):
                    print(f"    Branch {k+1}: '{branch.completion}' (score: {branch.score:.4f})")
            print()
    
    # Test different normalization scenarios
    test_cases = [
        {"normalize_by_prompt": True, "normalize_all_branches": False, "name": "Normalize branched completions by prompt"},
        {"normalize_by_prompt": False, "normalize_all_branches": False, "name": "Normalize branched completions globally"},
        {"normalize_by_prompt": True, "normalize_all_branches": True, "name": "Normalize all branches by prompt"},
        {"normalize_by_prompt": False, "normalize_all_branches": True, "name": "Normalize all branches globally"},
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {test_case['name']}")
        print(f"{'='*60}")
        
        # Create fresh test data for each test
        test_data = create_test_data()
        
        # First populate branched completion scores using average method
        _populate_brached_completion_scores(test_data, BranchedCompletionScoringMethod.AVERAGE)
        
        print_all_scores(test_data, "BEFORE NORMALIZATION")
        
        # Calculate stats before normalization
        if test_case["normalize_all_branches"]:
            mean_before, std_before = calculate_stats(test_data, "branch")
            print(f"Branch scores - Mean: {mean_before:.4f}, Std: {std_before:.4f}")
        else:
            mean_before, std_before = calculate_stats(test_data, "branched")
            print(f"Branched completion scores - Mean: {mean_before:.4f}, Std: {std_before:.4f}")
        
        # Apply normalization
        normalize_by_prompt_enum = NormalizeMethod.PROMPT if test_case["normalize_by_prompt"] else NormalizeMethod.ALL
        _normalize_scores(
            test_data,
            normalize_by_prompt=normalize_by_prompt_enum,
            normalize_prompt_ratio=0.5,
            normalize_all_branches=test_case["normalize_all_branches"]
        )
        
        print_all_scores(test_data, "AFTER NORMALIZATION")
        
        # Calculate stats after normalization
        if test_case["normalize_all_branches"]:
            mean_after, std_after = calculate_stats(test_data, "branch")
            print(f"Branch scores - Mean: {mean_after:.4f}, Std: {std_after:.4f}")
        else:
            mean_after, std_after = calculate_stats(test_data, "branched")
            print(f"Branched completion scores - Mean: {mean_after:.4f}, Std: {std_after:.4f}")
        
        print(f"\nChange in mean: {mean_before:.4f} -> {mean_after:.4f}")
        print(f"Change in std: {std_before:.4f} -> {std_after:.4f}")
