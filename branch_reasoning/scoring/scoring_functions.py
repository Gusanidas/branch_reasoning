import re
import random
from branch_reasoning.generation.completions import BranchedCompletion
from branch_reasoning.utils.utils import evaluate_expression

reasoning_start = "<think>"
reasoning_end = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"

format_keywords = [reasoning_start, reasoning_end, solution_start, solution_end]

main_format_regex_str = r"^[\s]*<think>(.*?)</think>.*?<answer>.*?</answer>[\s]*$"
main_format_regex = re.compile(main_format_regex_str, flags=re.MULTILINE | re.DOTALL)

loose_format_regex_str = r".*?<think>(.*?)</think>.*?<answer>.*?</answer>.*?$"
loose_format_regex = re.compile(loose_format_regex_str, flags=re.MULTILINE | re.DOTALL)

def _get_score_text(prompt: str, completion: str, use_vllm: bool = False): #TODO (change this)
    #return completion[len(prompt):]
    return completion

def match_format_exactly(branched_completion: BranchedCompletion, prompt: str, **kwargs):
    total_score = 0
    for branch in branched_completion.branches:
        score = 0
        text = _get_score_text(prompt, branch.completion)
        if main_format_regex.fullmatch(text) is not None:
            score += 2.0
        branch.score += score
        total_score += score
    return total_score/len(branched_completion.branches)

def match_format_loosely(branched_completion: BranchedCompletion, prompt: str, **kwargs):
    total_score = 0
    for branch in branched_completion.branches:
        score = 0
        text = _get_score_text(prompt, branch.completion)
        if loose_format_regex.fullmatch(text) is not None:
            score += 1.0
        branch.score += score
        total_score += score
    return total_score/len(branched_completion.branches)

def match_format_approximately(branched_completion: BranchedCompletion, prompt: str, **kwargs):
    total_score = 0
    for branch in branched_completion.branches:
        score = 2.0
        text = _get_score_text(prompt, branch.completion)
        for keyword in format_keywords:
            score += (
                0.5 if text.count(keyword) == 1 else -0.1 * (text.count(keyword)**0.5)
            )
        branch.score += max(0, score) # We want to avoid negative scores (branches)
        total_score += max(0, score)
    return total_score/len(branched_completion.branches)
    
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
        if random.random() < 0.01:
            print(f"Scoring Text first and last 20 characters: {text[:20]}...{text[-20:]}")
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

def get_branch_regex(branch_factor):
    """
    Generate a regex pattern for branch matching based on the branch factor.
    
    Args:
        branch_factor: Integer between 2 and 4 representing the number of branches.
        
    Returns:
        A compiled regex pattern object.
        
    Raises:
        ValueError: If branch_factor is not between 2 and 4.
    """
    if not 2 <= branch_factor <= 4:
        raise ValueError("Branch factor must be between 2 and 4 inclusive")
    
    # Define the branch patterns
    branches = ["a", "b", "c", "d"]
    branch_patterns = []
    option_patterns = []
    
    # Generate pattern for each branch up to branch_factor
    for i in range(branch_factor):
        branch = branches[i]
        pattern = f"<{branch}>(?:(?!<{branch}>|</{branch}>).){0,100}?</{branch}>"
        branch_patterns.append(pattern)
        option_patterns.append(f"#{branch}#")
    
    # Join patterns with whitespace
    combined_pattern = r"\s*".join(branch_patterns) + r"\s*(?:" + "|".join(option_patterns) + ")"
    
    # Compile and return the regex
    return re.compile(combined_pattern, flags=re.DOTALL)


def get_loose_branch_regex(branch_factor):
    """
    Generate a loose regex pattern for branch matching based on the branch factor.
    
    Args:
        branch_factor: Integer between 2 and 4 representing the number of branches.
        
    Returns:
        A compiled regex pattern object.
        
    Raises:
        ValueError: If branch_factor is not between 2 and 4.
    """
    if not 2 <= branch_factor <= 4:
        raise ValueError("Branch factor must be between 2 and 4 inclusive")
    
    # Define the branch patterns
    branches = ["a", "b", "c", "d"]
    branch_patterns = []
    option_patterns = []
    
    # Generate pattern for each branch up to branch_factor
    for i in range(branch_factor):
        branch = branches[i]
        pattern = f"<{branch}>.*?</{branch}>"
        branch_patterns.append(pattern)
        option_patterns.append(f"#{branch}#")
    
    # Join patterns with whitespace
    combined_pattern = r"\s*".join(branch_patterns) + r"\s*(?:" + "|".join(option_patterns) + ")"
    
    # Compile and return the regex
    return re.compile(combined_pattern, flags=re.DOTALL)

def _match_branch_fromat(text: str, max_branches: int, branch_regex: re.Pattern):
    matches = branch_regex.findall(text)
    return len(matches) if len(matches) <= max_branches else 0

def _match_loose_branch_fromat(text: str, max_branches: int, loose_branch_regex: re.Pattern):
    matches = loose_branch_regex.findall(text)
    return len(matches) if len(matches) <= max_branches else 0

# Regex to capture content between <think> and </think> tags
think_regex_str = r"<think>(.*?)</think>"
think_regex = re.compile(think_regex_str, flags=re.DOTALL)

def extract_thinking(text: str):
    match = think_regex.search(text)
    if match:
        return match.group(1).strip()
    return ""

def score_branch_format(branched_completion: BranchedCompletion, max_branches: int, branch_factor: int, prompt: str, **kwargs) -> float:
    """
    Score the text based on branch format matching.

    Args:
        text: The text to score
        max_branches: Maximum number of branches to consider
        prompt: The prompt to score
    Returns:
        float: Score based on branch format matching
    """
    total_score = 0
    branch_regex = get_branch_regex(branch_factor)
    for branch in branched_completion.branches:
        text = _get_score_text(prompt, branch.completion)
        thinking_text = extract_thinking(text)
        score = _match_branch_fromat(thinking_text, max_branches, branch_regex)
        branch.score += score
        total_score += score
    return total_score/len(branched_completion.branches)

def score_branch_format_loose(branched_completion: BranchedCompletion, max_branches: int, branch_factor: int, prompt: str, **kwargs) -> float:
    total_score = 0
    loose_branch_regex = get_loose_branch_regex(branch_factor)
    for branch in branched_completion.branches:
        text = _get_score_text(prompt, branch.completion)
        score = _match_loose_branch_fromat(text, max_branches, loose_branch_regex)
        branch.score += score
        total_score += score
    return total_score/len(branched_completion.branches)

_branch_keywords = [("<a>", 2), ("<b>", 2), ("<c>", 3), ("<d>", 4), ("</a>", 2), ("</b>", 2), ("</c>", 3), ("</d>", 4)]
_brach_choice_keyword = [("#a#", 2), ("#b#", 2), ("#c#", 3), ("#d#", 4)]

def _score_approx(text: str, branch_keywords: list[str], branch_choice_keyword: list[str], max_branches: int):
    score = 2.0
    for keyword in branch_keywords:
        count = text.count(keyword)
        if 0<=count <= max_branches:
            score += count * 0.2
        else:
            score -= 0.1 * (count**0.5)
    total_branch_choice_count = sum([text.count(keyword) for keyword in branch_choice_keyword])
    score += 1 if 0<=total_branch_choice_count <= max_branches else -1
    return max(0, score)

def score_branch_format_approx(branched_completion: BranchedCompletion, max_branches: int, branch_factor: int, prompt: str, **kwargs) -> float:
    branch_keywords = [keyword for keyword, n in _branch_keywords if n <= branch_factor]
    branch_choice_keyword = [keyword for keyword, n in _brach_choice_keyword if n <= branch_factor]
    total_score = 0
    for branch in branched_completion.branches:
        text = _get_score_text(prompt, branch.completion)
        thinking_text = extract_thinking(text)
        score = _score_approx(thinking_text, branch_keywords, branch_choice_keyword, max_branches)
        branch.score += score
        total_score += score
    return total_score/len(branched_completion.branches)


if __name__ == "__main__":
    text = "a b #a#"
    text_2 = "a b #a# c d #b#"
    text_3 = """There are many options open, I will branch out.
<a>
I will try to multiply 5 and 7, that is 5 * 7 = 35. And go from there.
</a>


<b>
I will multiply 7 and 3, that is 7 * 3 = 21. And go from there.
</b>

#a#
With 35, I can multiply by 2, that is 35 * 2 = 70.
Then I can subtract 3, that is 70 - 3 = 67.
That is the solution."""
    print(_match_branch_fromat(text))
    print(_match_branch_fromat(text_2))
    print(_match_branch_fromat(text_3))