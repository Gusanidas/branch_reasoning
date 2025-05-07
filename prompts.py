

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


single_branch_format_prompt = """
Place your reasoning between <think> and </think> tags.
Place your solution between <answer> and </answer> tags.
Dont write anything after </answer>.
"""


multi_branch_format_prompt= """
You can branch your reasoning. Describe briefly two different approaches.
Choose one of them by writing #a# or #b#. Describe both branches before choosing.
After branching, continue the reasoning of the branch you chose.
You can branch up to three times.
"""

# Examples
# Single branch

single_branch_example_1 = """
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

single_branch_example_2 = """
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

# Multi branch

multi_branch_example_1 = """
Numbers: 3, 5, 7, 2
Target: 67
<think>
There are many options open, I will branch out.
<a>
I will try to multiply 5 and 7, that is 5 * 7 = 35. And go from there.
</a>
<b>
I will multiply 7 and 3, that is 7 * 3 = 21. And go from there.
</b>
#a#
With 35, I can multiply by 2, that is 35 * 2 = 70.
Then I can subtract 3, that is 70 - 3 = 67.
That is the solution.
</think>

I have found the solution: 7*5*2 - 3 = 67.
<answer>7*5*2 - 3</answer>
"""

multi_branch_example_2 = """
Numbers: 9, 8, 7, 2
Target: 33
<think>
I will write different branches.
<a>
I will try multiplying 7 and 8, that is 7 * 8 = 56. This is too large for our target of 33.
</a>
<b>
I will try multiplying 9 and 2, that is 9 * 2 = 18. And go from there.
</b>
#b#
With 9 * 2 = 18, I need to add something to reach 33.
I need to add 15 more.
I have 7 and 8 remaining. If I add them: 7 + 8 = 15.
So, 18 + 15 = 33.
That is exactly our target!
</think>
I have found the solution: 9 * 2 + 7 + 8 = 33.
<answer>9 * 2 + 7 + 8</answer>
"""

multi_branch_example_3 = """
Numbers: 7, 12, 2, 20, 5
Target: 124
<think>
There are many options open, I will branch out.
<a>
I will try multiplying 7 and 20, that is 7 * 20 = 140.
</a>
<b>
I will try multiplying 7 and 12, that is 7 * 12 = 84. And go from there.
</b>
#b#
With 7 * 12 = 84, I need 30 more to reach 114.
I have 2, 20, and 5 remaining.
Let me see what combinations I can make. There are a few possibilities here.
Let me branch out again.
<a>
I will try adding 2 * 20 = 40 to our current value of 84.
84 + 40 = 124. This is exactly our target!
</a>
<b>
I will try subtracting 2 from our current value of 84.
84 - 2 = 82. This is too small.
</b>
#a#
This is the solution.
</think>
I have found the solution: 7 * 12 + 2 * 20 = 124.
<answer>7 * 12 + 2 * 20</answer>
"""

multi_branch_example_4 = """
Numbers: 7, 12, 2, 20, 5
Target: 124
<think>
I will start writing two branches.
<a>
I will try multiplying 7 and 12, that is 7 * 12 = 84.
</a>
<b>
I will try multiplying 7 and 20, that is 7 * 20 = 140.
</b>
#a#
With 7 * 12 = 84, I need 40 more to reach 124.
I have 2, 20, and 5 remaining.
Let me see what combinations I can make. There are a few possibilities here.
Let me branch out again.
<a>
I will try subtracting 2 from our current value of 84.
84 - 2 = 82. This is too small.
</a>
<b>
I will try adding 2 * 20 = 40 to our current value of 84.
84 + 40 = 124. This is exactly our target!
</b>
#b#
I have found the solution: 7 * 12 + 2 * 20 = 124.</think>
<answer>7 * 12 + 2 * 20</answer>"""

multi_branch_example_5 = """
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
I will start writing two branches.
<a>
I will try multiplying 7 and 12, that is 7 * 12 = 84.
</a>
<b>
I will try multiplying 7 and 20, that is 7 * 20 = 140.
</b>
#a#
With 7 * 12 = 84, I need 40 more to reach 124.
I have 2, 20, and 5 remaining.
Let me see what combinations I can make. There are a few possibilities here.
Let me branch out again.
<a>
I will try subtracting 2 from our current value of 84.
84 - 2 = 82.
</a>
<b>
I will try adding 2 * 20 = 40 to our current value of 84.
84 + 40 = 124.
</b>
#b#
From here, if I can add 9*5, that is 45, I will reach 169.
</think>
I have found the solution: 7 * 12 + 2 * 20 + 9 * 5 = 169.
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>"""

multi_branch_example_6 = """
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
I will start writing two branches.
<a>
I will try multiplying 7 and 20, that is 7 * 20 = 140.
</a>
<b>
I will try multiplying 7 and 12, that is 7 * 12 = 84. And go from there.
</b>
#b#
With 7 * 12 = 84, I need 40 more to reach 124.
I have 2, 20, and 5 remaining.
Let me see what combinations I can make. There are a few possibilities here.
Let me branch out again.
<a>
I will try subtracting 2 from our current value of 84.
84 - 2 = 82.
</a>
<b>
I will try adding 2 * 20 = 40 to our current value of 84.
84 + 40 = 124.
</b>
#b#
I have 7*12 + 2*20 = 124. And I have 5 and 9 remaining.
Let me see what combinations I can make. There are a few possibilities here.
Let me branch out again.
<a>
I will try multiplying 5 and 9, that is 5 * 9 = 45.
</a>
<b>
I will try adding 5 and 9, that is 5 + 9 = 14.
</b>
#a#
With 124, I can add 45, that is 124 + 45 = 169.
That is exactly our target!
</think>
I have found the solution: 7 * 12 + 2 * 20 + 9 * 5 = 169.
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>"""


multi_branch_example_7 = """
Numbers: 3, 5, 7, 2
Target: 67
<think>
<a>7*5</a> <b>7*3</b>
#a#
With 35, I can multiply by 2, that is 35 * 2 = 70.
Then I can subtract 3, that is 70 - 3 = 67.
</think>

<answer>7*5*2 - 3</answer>
"""
multi_branch_example_8 = """
Numbers: 3, 5, 7, 2
Target: 67
<think>
A: 7*3
B: 7*5
#b#
With 7*5, I can multiply by 2, that is 35 * 2 = 70.
Then I can subtract 3, that is 70 - 3 = 67.
</think>

<answer>7*5*2 - 3</answer>
"""
multi_branch_example_9 = """
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
A: 7*20
B: 7*12
#b#
With 7 * 12 = 84.
I have 2, 20, and 5 remaining.
A: 84 -2
B: 84 + 40
#b#
A: 9 * 5
B: 9 + 5
#a#
45 + 124 = 169.
</think>
I have found the solution: 7 * 12 + 2 * 20 + 9 * 5 = 169.
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>"""

multi_branch_example_10 = """
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
A: 7*12
B: 7*20
#a#
With 7 * 12 = 84.
I have 2, 20, and 5 remaining.
A: 84 *5
B: 84 + 40
124, left 9 and 5.
#b#
A: 9 * 5 + 124
B: 9 + 5 + 124
#a#
45 + 124 = 169.
</think>
I have found the solution: 7 * 12 + 2 * 20 + 9 * 5 = 169.
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>"""

multi_branch_example_10 = """
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
<a>7*12</a> <b>7*20</b>
#a#
With 7 * 12 = 84.
I have 2, 20, and 5 remaining.
<a>84 *5</a> <b>84 + 40</b>
124, left 9 and 5.
#b#
<a>9 * 5 + 124</a> <b>9 + 5 + 124</b>
#a#
45 + 124 = 169.
</think>
I have found the solution: 7 * 12 + 2 * 20 + 9 * 5 = 169.
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>"""

single_branch_examples = [single_branch_example_1, single_branch_example_2]
multi_branch_examples = [multi_branch_example_1, multi_branch_example_2, multi_branch_example_3, multi_branch_example_4, multi_branch_example_5, multi_branch_example_6, multi_branch_example_7, multi_branch_example_8, multi_branch_example_9, multi_branch_example_10]