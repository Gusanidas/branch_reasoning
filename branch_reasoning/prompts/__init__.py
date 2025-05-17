from branch_reasoning.prompts.branch_factor_2 import get_format_and_examples as get_format_and_examples_branch_factor_2
from branch_reasoning.prompts.branch_factor_3 import get_format_and_examples as get_format_and_examples_branch_factor_3
from branch_reasoning.prompts.branch_factor_4 import get_format_and_examples as get_format_and_examples_branch_factor_4
from branch_reasoning.prompts.no_branches import get_format_and_examples as get_format_and_examples_no_branches

base_prompt = """Using a list of numbers, find a mathematical expression that equals the target.
You can use addition (+), subtraction (-) and multiplication (*). Do not use division (/).
You must use each number at most once, and you don't have to use all numbers.
The numbers cannot be concatenated (e.g., you cannot use 9 and 5 to make 95).
Write just the left side of the expression.
{format_prompt}

Think about the solution and write it step by step.
{example}
Task:

Numbers: {nums}
Target: {target}
"""

def get_format_and_examples(branch_completions, max_branching_points, branching_factor):
    if branch_completions:
        if branching_factor == 2:
            return get_format_and_examples_branch_factor_2(max_branching_points)
        elif branching_factor == 3:
            return get_format_and_examples_branch_factor_3(max_branching_points)
        elif branching_factor == 4:
            return get_format_and_examples_branch_factor_4(max_branching_points)
    else:
        return get_format_and_examples_no_branches()
        
system_prompt = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""