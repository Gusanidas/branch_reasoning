from typing import List, Optional, Callable
from datasets import Dataset as HFDataset
import random

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def transform_countdown_data(
    input_dataset: HFDataset,
    base_prompt_template: str,
    template_func: Callable[[str], str] = None,
    format_prompt: str = "",
    examples: List[str] = [""], 
) -> HFDataset:
    """
    Transforms a countdown dataset (with numbers, target, solution) into a
    prompt-based format (with question, answer).

    Applies a base prompt template and a template function to each example.

    Args:
        input_dataset: A Hugging Face dataset with columns 'numbers' (List[int]),
                       'target' (int), 'solution' (str), and optionally 'tag' (str).
        base_prompt_template: A format string for the question, expecting
                              {nums} and {target} placeholders.
                              Example: "Use the numbers {nums} to reach the target {target}."
        template_func: A function that takes the formatted prompt string and
                       applies a final template (e.g., adding user/assistant roles).
                       Defaults to `apply_r1_template`.
        examples: A list of examples to be used as examples in the prompt.

    Returns:
        Dataset: A Hugging Face dataset with columns 'question' (str),
                 'numbers' (List[int]), 'target' (int), 'answer' (str),
                 and 'tag' (str) if present in the input.
    """
    if template_func is None:
        # Identity function
        template_func = lambda x: x

    def _format_task(task):
        """Internal function to format a single row."""
        example = random.choice(examples)
        try:
            formatted_prompt = base_prompt_template.format(
                nums=task['numbers'],
                target=task['target'],
                example=example, 
                format_prompt=format_prompt,
            )
            question = template_func(formatted_prompt)
            # Without examples
            bare_question = base_prompt_template.format(
                nums=task['numbers'],
                target=task['target'],
                example="", 
                format_prompt=format_prompt,
            )
            return {
                'question': question,
                #'bare_question': bare_question,
                'solution': task['solution']
            }
        except Exception as e:
            print(f"Error formatting prompt for task {task}:\n {e}")
            return {'question': '', 'answer': ''}

    transformed_dataset = input_dataset.map(
        _format_task,
    )

    expected_cols = {'question', 'numbers', 'target', 'solution', 'bare_question'}
    missing_cols = expected_cols - set(transformed_dataset.column_names)
    if 'tag' in input_dataset.column_names:
         expected_cols.add('tag')
         
    if missing_cols:
        print(f"Warning: Missing expected columns in transformed dataset: {missing_cols}")

    unexpected_cols = set(transformed_dataset.column_names) - expected_cols
    if unexpected_cols:
        print(f"Info: Found additional columns in transformed dataset: {unexpected_cols}")

    print(f"Final dataset columns: {transformed_dataset.column_names}")

    return transformed_dataset