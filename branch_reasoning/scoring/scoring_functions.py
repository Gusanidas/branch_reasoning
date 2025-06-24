import re
import random
from branch_reasoning.generation.completions import BranchedCompletion, Branch, ScoringData
from branch_reasoning.utils.utils import evaluate_expression
from collections import defaultdict


reasoning_start = "<think>"
reasoning_end = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"

format_keywords = [reasoning_start, reasoning_end, solution_start, solution_end]

main_format_regex_str = r"^[\s]*<think>(.*?)</think>.*?<answer>.*?</answer>[\s]*$"
main_format_regex = re.compile(main_format_regex_str, flags=re.MULTILINE | re.DOTALL)

main_format_regex_str = r"^[\s]*<think>((?:(?!<think>|</think>).)*?)</think>((?:(?!</think>).)*?)<answer>((?:(?!<answer>|</answer>).)*?)</answer>[\s]*$"
main_format_regex = re.compile(main_format_regex_str, flags = re.MULTILINE | re.DOTALL)

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
        branch.meta_scores["match_format_exactly"] += score
        branch.meta_scores["match_format_exactly_steps"] += 1
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
        branch.meta_scores["match_format_loosely"] += score
        branch.meta_scores["match_format_loosely_steps"] += 1
        total_score += score
    return total_score/len(branched_completion.branches)

def match_format_approximately(branched_completion: BranchedCompletion, prompt: str, **kwargs):
    total_score = 0
    for branch in branched_completion.branches:
        score = 3.0
        text = _get_score_text(prompt, branch.completion)
        for keyword in format_keywords:
            score += (
                0.5 if text.count(keyword) == 1 else -0.4 * (text.count(keyword)**0.5)
            )
        branch.score += max(0, score) # We want to avoid negative scores (branches)
        branch.meta_scores["match_format_approximately"] += max(0, score)
        branch.meta_scores["match_format_approximately_steps"] += 1
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
        score += 3 if abs(result - target) < 0.0001 else 0
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
            branch.meta_scores["countdown_answer"] += score
            branch.meta_scores["countdown_answer_steps"] += 1
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
            branch.meta_scores["countdown_answer"] += score
            branch.meta_scores["countdown_answer_steps"] += 1
    elif mode == "avg":
        score = sum(scores) / len(scores)
        for branch in branched_completion.branches:
            branch.score += score
            branch.meta_scores["countdown_answer"] += score
            branch.meta_scores["countdown_answer_steps"] += 1
    elif mode == "eq":
        score = sum(scores)/len(scores)
    return score


def _rate_countdown_answer_individual(
    completion,
    nums,
    target,
):
    match = re.search(f"<answer>(.*?)</answer>", completion)
    score = 0.0
    if not match:
        return score
    extracted_answer = match.group(1).strip()
    if len(extracted_answer) <1:
        return 0.25
    try:
        used_numbers = [int(n) for n in re.findall(r"\d+", extracted_answer)]
        nums_counter = {}
        for n in nums:
            nums_counter[n] = nums_counter.get(n, 0) + 1
        for n in used_numbers:
            if n not in nums_counter or nums_counter[n] == 0:
                return 0.0
            nums_counter[n] -= 1
        result = evaluate_expression(extracted_answer)
        score += 1 if abs(result - target) < 0.001 else 0
        return score
    except Exception as e:
        return 0.0

def rate_countdown_answer_individual(
    branched_completion: BranchedCompletion,
    prompt: str,
    scoring_data: dict,
    use_ratio: bool = True,
    total_score_sum: float = 3,
    **kwargs,
):
    total_score = 0
    for branch in branched_completion.branches:
        score = _rate_countdown_answer_individual(branch.completion, scoring_data.nums, scoring_data.target)
        total_score += score
    ratio = (len(branched_completion.branches) * total_score_sum)/(total_score + 1e-3) if use_ratio else 1
    for branch in branched_completion.branches:
        score = _rate_countdown_answer_individual(branch.completion, scoring_data.nums, scoring_data.target)
        branch.score += score*ratio
        branch.meta_scores["countdown_answer_individual"] += score*ratio
        branch.meta_scores["countdown_answer_individual_steps"] += 1
    return total_score

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
        interval = '{0,100}'
        pattern = f"<{branch}>(?:(?!<{branch}>|</{branch}>).){interval}?</{branch}>"
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
    return 0.5*(len(matches)**0.2) if len(matches) <= max_branches else 0

def _match_loose_branch_fromat(text: str, max_branches: int, loose_branch_regex: re.Pattern):
    matches = loose_branch_regex.findall(text)
    return 0.2*(len(matches)**0.2) if len(matches) <= max_branches else 0

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
        branch.meta_scores["branch_format"] += score
        branch.meta_scores["branch_format_steps"] += 1
        total_score += score
    return total_score/len(branched_completion.branches)

def score_branch_format_loose(branched_completion: BranchedCompletion, max_branches: int, branch_factor: int, prompt: str, **kwargs) -> float:
    total_score = 0
    loose_branch_regex = get_loose_branch_regex(branch_factor)
    for branch in branched_completion.branches:
        text = _get_score_text(prompt, branch.completion)
        thinking_text = extract_thinking(text)  
        score = _match_loose_branch_fromat(thinking_text, max_branches, loose_branch_regex)
        branch.score += score
        branch.meta_scores["branch_format_loose"] += score
        total_score += score
    return total_score/len(branched_completion.branches)

_branch_keywords = [("<a>", 2), ("<b>", 2), ("<c>", 3), ("<d>", 4), ("</a>", 2), ("</b>", 2), ("</c>", 3), ("</d>", 4)]
_brach_choice_keyword = [("#a#", 2), ("#b#", 2), ("#c#", 3), ("#d#", 4)]

def _score_approx(text: str, branch_keywords: list[str], branch_choice_keyword: list[str], max_branches: int):
    score = 2.0
    branch_keywords_count = [text.count(keyword) for keyword in branch_keywords]
    score += sum([0.5*(count) if 0<=count <= max_branches else -0.3*(count) for count in branch_keywords_count])
    total_branch_choice_count = sum([text.count(keyword) for keyword in branch_choice_keyword])
    score += 1 if 0<=total_branch_choice_count <= max_branches else -1

    for count in branch_keywords_count:
        for o_count in branch_keywords_count + [total_branch_choice_count]:
            if count != o_count:
                score -= 1
    return max(0, score*0.1)

def _score_outside_text(text: str, branch_keywords: list[str], branch_choice_keyword: list[str]):
    score = 2.0
    for keyword in branch_keywords + branch_choice_keyword:
        if keyword in text:
            score -= 0.1*(text.count(keyword))
    return max(0, score*0.1)

def score_branch_format_approx(branched_completion: BranchedCompletion, max_branches: int, branch_factor: int, prompt: str, **kwargs) -> float:
    branch_keywords = [keyword for keyword, n in _branch_keywords if n <= branch_factor]
    branch_choice_keyword = [keyword for keyword, n in _brach_choice_keyword if n <= branch_factor]
    total_score = 0
    for branch in branched_completion.branches:
        text = _get_score_text(prompt, branch.completion)
        thinking_text = extract_thinking(text)
        score = _score_approx(thinking_text, branch_keywords, branch_choice_keyword, max_branches)
        score += _score_outside_text(text, branch_keywords, branch_choice_keyword)
        branch.score += score
        branch.meta_scores["branch_format_approx"] += score
        branch.meta_scores["branch_format_approx_steps"] += 1
        total_score += score
    return total_score/len(branched_completion.branches)

def score_by_length(branched_completion: BranchedCompletion, prompt: str, **kwargs) -> float:
    """
    Score based on text length, giving 2 points for length 0 and diminishing exponentially to 0 for ~2500 characters.
    
    Args:
        branched_completion: The branched completion to score
        prompt: The prompt (used to get scoring text)
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Average score across all branches
    """
    total_score = 0
    for branch in branched_completion.branches:
        text = _get_score_text(prompt, branch.completion)
        length = len(text)
        
        decay_constant = 1000
        score = 1.0 * (2.71828 ** (-length / decay_constant))
        
        branch.score += score
        branch.meta_scores["length_score"] += score
        branch.meta_scores["length_score_steps"] += 1
        total_score += score
    
    return 1#total_score / len(branched_completion.branches)

if __name__ == "__main__":
    text = "a b #a#"
    text_2 = "a b #a# c d #b#"
    text_3 = """

Numbers: [10, 6]
Target: 16
Think about the solution and write it step by step.

<think>
There is a straightforward approach to consider first.
I will start with trying to multiply 10 and 6 together since it's simple and gives a larger base value.
If we get 10 * 6 = 60, we need to find how to reach 16 using the remaining number, which is 0 in this case.
Since we cannot use numbers together (like 10 and 6 forming 106 to get 106), subtracting or adding 0 will not help.
Instead, I'll explore adding or subtracting the smaller number to the product.
Let me see how this plays out.
<a>
First attempt: I'll add 6 to 10 * 6 = 60.
60 + 6 = 66.
</a>
<b>
Second attempt: Instead of adding, I'll try subtracting 6 from 10 * 6 = 60.
60 - 6 = 54.
</b>
#b#

#b#
Neither of these paths led us closer to our target of 16.
Next, I'll consider if any combination of addition or subtraction of the smaller number (0) to 10 * 6 can bring us close.
However, since we cannot use 0, let me think about using the smaller number directly in combination.
Another way could be checking the possibility of using 10 itself without multiplying, then see if we can achieve 16.
Since using 10 alone does not help (10 itself is far away from 16), let me think if combining operations can make a difference.
But since multiplication and addition/subtraction are limited to 10 and 6, I've exhausted possible straightforward combinations.
So, I might need to look at a more indirect method or realize I might need more numbers to work with here, but as given, the available set seems insufficient.

This reasoning has not led to a viable solution. Let's consider an alternative.
</think>
No solution with the given numbers and operations is apparent. Since I haven't found a valid expression, I need to indicate that:


<answer></answer>
"""
    text_6 = """

Numbers: [10, 6]
Target: 16
Think about the solution and write it step by step.

<think>
There is a straightforward approach to consider first.
I will start with trying to multiply 10 and 6 together since it's simple and gives a larger base value.
If we get 10 * 6 = 60, we need to find how to reach 16 using the remaining number, which is 0 in this case.
Since we cannot use numbers together (like 10 and 6 forming 106 to get 106), subtracting or adding 0 will not help.
Instead, I'll explore adding or subtracting the smaller number to the product.
Let me see how this plays out.
Neither of these paths led us closer to our target of 16.
Next, I'll consider if any combination of addition or subtraction of the smaller number (0) to 10 * 6 can bring us close.
However, since we cannot use 0, let me think about using the smaller number directly in combination.
Another way could be checking the possibility of using 10 itself without multiplying, then see if we can achieve 16.
Since using 10 alone does not help (10 itself is far away from 16), let me think if combining operations can make a difference.
But since multiplication and addition/subtraction are limited to 10 and 6, I've exhausted possible straightforward combinations.
So, I might need to look at a more indirect method or realize I might need more numbers to work with here, but as given, the available set seems insufficient.

This reasoning has not led to a viable solution. Let's consider an alternative.
</think>
No solution with the given numbers and operations is apparent. Since I haven't found a valid expression, I need to indicate that:

<answer></answer>
"""

    text_3 ="<think>\nI will start writing two branches.\n<a>\nI will try multiplying 9 and 5, that is 9 * 5 = 45.\n</a>\n<b>\nI will try multiplying 9 and 22, that is 9 * 22 = 198.\n</b>\n#a#\n\n\nWith 9 * 5 = 45, I need 9 more to reach 54.\nThe remaining numbers are 22.\nLet me see what combinations I can make.\n<a>\nI will try subtracting 22 from 45.\n45 - 22 = 23.\n</a>\n<b>\nI will try adding 22 to 45.\n45 + 22 = 67.\n</b>\n#b#\nFrom here, if I can add 9*22, that is 198, I will exceed the target, so this doesn't work.\nI can branch again with the first approach where we have already found a valid combination.\n</think>\n\nSince the second branch did not lead us to a solution, let's consider the first approach which is correct:\n9 * 5 + 22 - 23 = 36.\n<answer>9 * 5 + 22 - 23</answer>"

    text_4 = """
<think>
I will branch out my approach.
With 35, it is too high to reach 12 with any operation, so I will try adding the remaining number.
With 12, I need to see if I can reach this using the operations.

I have found the solution: 5 + 7 = 12.

With 12, it is already the target, so this works directly.
</think>

I have found the solution: 5 + 7 = 12.
<answer>5 + 7</answer>
"""


    branch = Branch(completion=text_4, log_probs=None, ref_log_probs=None, score=0, key="test", meta_scores=defaultdict(float))
    branched_completion = BranchedCompletion(branches=[branch], score=0)
    print("Branching format")
    print(f"Approx")
    print(score_branch_format_approx(branched_completion, 2, 2, text_3))
    print(f"exact")
    print(score_branch_format(branched_completion, 2, 2, text_3))
    print(f"loose")
    print(score_branch_format_loose(branched_completion, 2, 2, text_3))
    print(f"rate_countdown_answer")
    scoring_data = ScoringData(nums=[9, 5, 22], target=36)
    print(rate_countdown_answer(branched_completion, text_3, scoring_data=scoring_data, wandb_logs=defaultdict(float)))
    print("================================================")
    print(f"Format")
    print(f"Approx")
    print(match_format_approximately(branched_completion, text_3))
    print(f"exact")
    print(match_format_exactly(branched_completion, text_3))
    print(f"loose")
    print(match_format_loosely(branched_completion, text_3))


    print(f"Brach regex {get_branch_regex(2)}")