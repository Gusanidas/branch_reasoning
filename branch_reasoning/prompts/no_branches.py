

format_prompt = """
Place your reasoning between <think> and </think> tags.
Place your solution between <answer> and </answer> tags.
Dont write anything after </answer>.
"""

example_1 = """
Numbers: 3, 5, 7
Target: 22
<think>
I will try to multiply 5 and 7, that is 5 * 7 = 35. Minus 3 that is 32, which is too much.
I can try adding all the numbers. 3 + 5 + 7 = 15, it is too low.
I can multiply 3 and 5, that is 3 * 5 = 15. I can add 7, that is 15 + 7 = 22.
That is the solution.
</think>

I have found the solution: 3*5 + 7 = 22.
<answer>3 * 5 + 7</answer>"""

example_2 = """
Numbers: 4, 5, 9
Target: 20
<think>
First, I'll try adding all numbers together: 4 + 5 + 9 = 18. This is close but still 2 less than our target.
I could try multiplication. 4 * 5 = 20, which is exactly our target! Great, I found a solution.
Let me check if there are other solutions.
I could try 5 * 9 = 45, and then subtract 4: 45 - 4 = 41. That's too large.
Another option would be 4 * 9 = 36, and then subtract 5: 36 - 5 = 31. Also too large.
So the best solution is 4 * 5 = 20.
</think>
I have found the solution: 4 * 5 = 20.
<answer>4 * 5</answer>
"""

examples = [example_1, example_2]
def get_format_and_examples():
    return format_prompt, examples