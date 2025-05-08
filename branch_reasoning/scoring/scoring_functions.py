import re
from branch_reasoning.generation.completions import BranchedCompletion
from branch_reasoning.utils.utils import evaluate_expression

reasoning_start = "<think>"
reasoning_end = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"

format_keywords = [reasoning_start, reasoning_end, solution_start, solution_end]

main_format_regex_str = r"^[\s]*<think>(.*?)</think>.*?<answer>.*?</answer>[\s]*$"
main_format_regex = re.compile(main_format_regex_str, flags=re.MULTILINE | re.DOTALL)

def _get_score_text(prompt: str, completion: str):
    return completion[len(prompt)-8:]

def match_format_exactly(branched_completion: BranchedCompletion, prompt: str, **kwargs):
    total_score = 0
    for branch in branched_completion.branches:
        score = 0
        text = _get_score_text(prompt, branch.completion)
        if main_format_regex.fullmatch(text) is not None:
            score += 2.0
        branch.score += score
        total_score += score
    return total_score


def match_format_approximately(branched_completion: BranchedCompletion, prompt: str, **kwargs):
    total_score = 0
    for branch in branched_completion.branches:
        score = 0
        text = _get_score_text(prompt, branch.completion)
        for keyword in format_keywords:
            score += (
                0.5 if text.count(keyword) == 1 else -0.1 * (text.count(keyword)**0.5)
            )
        branch.score += score
        total_score += score
    return total_score
    
def _rate_countdown_answer_strict(
    completion,
    nums,
    target,
):
    match = re.search(f"<answer>(.*?)</answer>", completion)
    score = 0.0
    if not match:
        return score
    extracted_answer = match.group(1).strip()
    try:
        used_numbers = [int(n) for n in re.findall(r"\d+", extracted_answer)]
        nums_counter = {}
        for n in nums:
            nums_counter[n] = nums_counter.get(n, 0) + 1
        for n in used_numbers:
            if n not in nums_counter or nums_counter[n] == 0:
                return score
            nums_counter[n] -= 1
        result = evaluate_expression(extracted_answer)
        score += 5 if abs(result - target) < 0.0001 else 0
        return score
    except Exception as e:
        return score

def rate_countdown_answer(
    branched_completion: BranchedCompletion,
    prompt: str,
    scoring_data: dict,
    mode = "max", # "max", "avg", "eq"
    wandb_logs = None,
    **kwargs,
):
    scores = []
    for branch in branched_completion.branches:
        text = _get_score_text(prompt, branch.completion)
        score = _rate_countdown_answer_strict(text, scoring_data.nums, scoring_data.target)
        if mode == "eq":
            branch.score += score
        scores.append(score)
    wandb_logs["score/countdown_max"] += max(scores)
    wandb_logs["score/countdown_min"] += min(scores)
    wandb_logs["score/countdown_avg"] += sum(scores) / len(scores)
    wandb_logs["score/countdown_max_steps"] += 1
    wandb_logs["score/countdown_min_steps"] += 1
    wandb_logs["score/countdown_avg_steps"] += 1
    if mode == "max":
        score = max(scores)
        for branch in branched_completion.branches:
            branch.score += score
    elif mode == "avg":
        score = sum(scores) / len(scores)
        for branch in branched_completion.branches:
            branch.score += score
    elif mode == "eq":
        score = sum(scores)/len(scores)
    return score