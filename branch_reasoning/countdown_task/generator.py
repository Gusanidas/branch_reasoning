from typing import List, Optional
import operator
import time
import random
from itertools import combinations, permutations, combinations_with_replacement
import pandas as pd
from datasets import Dataset as HFDataset, concatenate_datasets
from branch_reasoning.utils.utils import evaluate_expression

def find_solution(numbers: List[int], target: int) -> Optional[str]:
    """
    Find a mathematical expression using the given numbers that equals the target.
    Returns None if no solution is found.
    """
    operators = {'+': operator.add, '-': operator.sub, '*': operator.mul}
    
    # Try different number of operands from 2 to len(numbers)
    for n_operands in range(2, len(numbers) + 1):
        # Try different combinations of numbers
        for nums in combinations(numbers, n_operands):
            # Try different orderings of the numbers
            for nums_perm in permutations(nums):
                # Try different combinations of operators
                for ops in combinations_with_replacement(list(operators.keys()), n_operands - 1):
                    expr = str(nums_perm[0])
                    for i, op in enumerate(ops):
                        expr += f" {op} {nums_perm[i + 1]}"
                    
                    result = evaluate_expression(expr)
                    if abs(result - target) < 0.0001:  # Account for floating point precision
                        return expr
    
    return None

def get_countdown_tasks_data(
    n: int = 1000,
    min_num: int = 1, max_num: int = 100,
    min_numbers: int = 4, max_numbers: int = 6,
    min_target: int = 100, max_target: int = 999,
    tag: Optional[str] = None
) -> HFDataset:
    """
    Generate a Hugging Face dataset for the countdown task with numbers, target, and solution.

    Generates 'n' examples where a solution is found using the provided find_solution function.

    Args:
        n: Number of solvable examples to generate.
        min_num/max_num: Range for the numbers to use.
        min_numbers/max_numbers: Range for how many numbers to provide.
        min_target/max_target: Range for the target number.
        tag: Optional tag to add as a column to each row in the dataset.

    Returns:
        Dataset: A Hugging Face dataset with columns: 'numbers' (List[int]),
                 'target' (int), 'solution' (str), and optionally 'tag' (str).
    """
    numbers_list = []
    targets = []
    solutions = []

    while len(solutions) < n:
        num_count = random.randint(min_numbers, max_numbers)
        nums = [random.randint(min_num, max_num) for _ in range(num_count)]
        target = random.randint(min_target, max_target)

        solution = find_solution(nums, target)

        if solution:
            numbers_list.append(nums)
            targets.append(target)
            solutions.append(solution)

    df_data = {
        "numbers": numbers_list,
        "target": targets,
        "solution": solutions,
    }

    if tag is not None:
        df_data["tag"] = [tag] * len(solutions)

    df = pd.DataFrame(df_data)
    dataset = HFDataset.from_pandas(df)
    return dataset

def make_combined_countdown_tasks(
    very_easy: int = 0,
    easy: int = 0,
    medium: int = 0,
    hard: int = 0,
    very_hard: int = 0,
    shuffle_result: bool = True,
) -> HFDataset:
    """
    Generates and combines countdown task datasets of varying difficulties.

    Calls get_countdown_tasks_data for each specified difficulty level and
    concatenates them into a single Hugging Face dataset.

    Args:
        very_easy: Number of 'very easy' examples to generate.
        easy: Number of 'easy' examples to generate.
        medium: Number of 'medium' examples to generate.
        hard: Number of 'hard' examples to generate.
        very_hard: Number of 'very hard' examples to generate.
        shuffle_result: If True, shuffle the final combined dataset.

    Returns:
        Dataset: A combined Hugging Face dataset containing examples from
                 all requested difficulty levels.
    """
    all_datasets = []

    if very_easy > 0:
        t0 = time.time()
        very_easy_dataset = get_countdown_tasks_data(
            n=very_easy,
            min_num=5,
            max_num=10,
            min_numbers=2,
            max_numbers=2,
            min_target=10,
            max_target=20,
            tag="very_easy",
        )
        all_datasets.append(very_easy_dataset)
        print(f"Time to generate {very_easy} very easy examples: {time.time() - t0:.2f}s")

    if easy > 0:
        t0 = time.time()
        easy_dataset = get_countdown_tasks_data(
            n=easy,
            min_num=2,
            max_num=22,
            min_numbers=3,
            max_numbers=3,
            min_target=4,
            max_target=100,
            tag="easy",
        )
        all_datasets.append(easy_dataset)
        print(f"Time to generate {easy} easy examples: {time.time() - t0:.2f}s")

    if medium > 0:
        t0 = time.time()
        medium_dataset = get_countdown_tasks_data(
            n=medium,
            min_num=1,
            max_num=100,
            min_numbers=4,
            max_numbers=4,
            min_target=1,
            max_target=500,
            tag="medium",
        )
        all_datasets.append(medium_dataset)
        print(f"Time to generate {medium} medium examples: {time.time() - t0:.2f}s")

    if hard > 0:
        t0 = time.time()
        hard_dataset = get_countdown_tasks_data(
            n=hard,
            min_num=1,
            max_num=100,
            min_numbers=5,
            max_numbers=5,
            min_target=1,
            max_target=2_000,
            tag="hard",
        )
        all_datasets.append(hard_dataset)
        print(f"Time to generate {hard} hard examples: {time.time() - t0:.2f}s")

    if very_hard > 0:
        t0 = time.time()
        very_hard_dataset = get_countdown_tasks_data(
            n=very_hard,
            min_num=1,
            max_num=80,
            min_numbers=6,
            max_numbers=7,
            min_target=1,
            max_target=10_000,
            tag="very_hard",
        )
        all_datasets.append(very_hard_dataset)
        print(f"Time to generate {very_hard} very hard examples: {time.time() - t0:.2f}s")

    if not all_datasets:
        print("Warning: No examples requested for any difficulty level. Returning empty dataset.")
        empty_df = pd.DataFrame({
             "numbers": pd.Series(dtype='object'),
             "target": pd.Series(dtype='int'),
             "solution": pd.Series(dtype='str'),
             "tag": pd.Series(dtype='str')
         })
        combined_dataset = HFDataset.from_pandas(empty_df)
        return combined_dataset


    print(f"Concatenating {len(all_datasets)} dataset(s)...")
    combined_dataset = concatenate_datasets(all_datasets)

    if shuffle_result:
        print("Shuffling the combined dataset...")
        combined_dataset = combined_dataset.shuffle()

    print(f"Final combined dataset size: {len(combined_dataset)} examples.")
    return combined_dataset


if __name__ == "__main__":
    # Test code
    from branch_reasoning.prompts.prompts import base_prompt
    from branch_reasoning.countdown_task.templates import apply_r1_template, transform_countdown_data
    
    print("Generating a small 'easy' dataset...")
    small_dataset = make_combined_countdown_tasks(easy=5, shuffle_result=False)

    print("\n--- Original Small Dataset ---")
    for i, example in enumerate(small_dataset):
        print(f"Example {i+1}:")
        for k, v in example.items():
            print(f"  {k}: {v}")
        print("-" * 10)

    print("\nTransforming the dataset...")
    transformed_dataset = transform_countdown_data(
        input_dataset=small_dataset,
        base_prompt_template=base_prompt,
        template_func=apply_r1_template,
    )

    print("\n--- Transformed Dataset ---")
    for i, example in enumerate(transformed_dataset):
        print(f"Example {i+1}:")
        print(f"  Question: {example.get('question', 'N/A')}")
        print(f"  Solution: {example.get('solution', 'N/A')}")
        print(f"  Original Numbers: {example.get('numbers', 'N/A')}")
        print(f"  Original Target: {example.get('target', 'N/A')}")
        print(f"  Tag: {example.get('tag', 'N/A')}")
        print("-" * 10)

    print("\nScript finished.")