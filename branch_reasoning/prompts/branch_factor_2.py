multi_branch_format_prompt= """
Place your reasoning between <think> and </think> tags.
Place your solution between <answer> and </answer> tags.
Dont write anything after </answer>.

You can branch your reasoning. Describe briefly two different approaches.
And follow that reasoning path, another instance will follow the other path.
In between <think> and </think>, write one branch between <a> and </a>, and the other between <b> and </b>.
Choose one of them by writing #a# or #b#.
So, for example:
<a>
Short description of the branch.
</a>
<b>
Short description of the branch.
</b>
#a#

After branching, continue the reasoning only of the branch you chose.
If the branch leads to no solution, write:
<answer></answer>
You can branch up to {max_branching_points} times.
"""


example_1 = ("""
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
""", 1)

example_2 = ("""
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
""", 1)

example_3 = ("""
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
""", 2)

example_4 = ("""
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
<answer>7 * 12 + 2 * 20</answer>""", 2)

example_5 = ("""
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
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>""", 2)

example_6 = ("""
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
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>""",3)


example_7 = ("""
Numbers: 3, 5, 7, 2
Target: 67
<think>
<a>7*5</a> <b>7*3</b>
#a#
With 35, I can multiply by 2, that is 35 * 2 = 70.
Then I can subtract 3, that is 70 - 3 = 67.
</think>

<answer>7*5*2 - 3</answer>
""", 1)

example_8 = ("""
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
Branching out.
<a>7*20</a> <b>7*12</b>
#b#
40 left to reach 169.
2, 20, and 5 remaining.
<a>
84 - 2 = 82.
</a>
<b>
84 + 40 = 124.
</b>
#b#
I have 7*12 + 2*20 = 124. And I have 5 and 9 remaining.
<a>5*9 = 45</a> <b>5+9 = 14</b>
#a#
With 124, I can add 45, that is 124 + 45 = 169.
That is exactly our target!
</think>
I have found the solution: 7 * 12 + 2 * 20 + 9 * 5 = 169.
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>""",3)

example_10 = ("""
Numbers: 23,49, 5, 85, 26
Target: 41
<think>
Branching out.
<a>23 + 49</a> <b>23 + 5</b>
#a#

23 + 49 = 72
<a>85-72 = 13</a> <b>72 - 26 = 46</b>
#b#

So, 23 + 49 - 26 = 41.

46 - 5 = 41
</think>
I have found the solution: 23 + 49 - 26 - 5 = 41.
<answer>23 + 49 - 26 - 5</answer>
""",2)


example_11 = ("""
Numbers: 23,49, 5, 85, 26
Target: 82
<think>
Branching out.
<a>26 + 49 = 75</a> <b>26 - 23 = 3</b>
#b#

85 - 3 = 82.

</think>
I have found the solution: 85 + 26 - 23 = 82.
<answer>85 + 26 - 23</answer>
""",1)

example_12 = ("""
Numbers: 20, 44, 50, 10, 12, 18
Target: 92
<think>
Branching out.
<a>20 + 44</a> <b>20 + 50</b>
#a#

20 + 44 = 64
<a>64 + 10 = 74</a> <b>64 + 12 = 76</b>
#a#

74 + 12 = 86
74 + 18 = 92.

20 + 44 + 10 + 18 = 92.
</think>
I have found the solution: 20 + 44 + 10 + 12 + 18 = 92.
<answer>20 + 44 + 10 + 12 + 18</answer>""",2)


no_branches_example_1 = ("""
Numbers: 3, 5, 7
Target: 22
<think>
I will try to multiply 5 and 7, that is 5 * 7 = 35. Minus 3 that is 32, which is too much.
</think>
I can try adding all the numbers. 3 + 5 + 7 = 15, it is too low.
I can multiply 3 and 5, that is 3 * 5 = 15. I can add 7, that is 15 + 7 = 22.
That is the solution.
</think>

I have found the solution: 3*5 + 7 = 22.
<answer>3 * 5 + 7</answer>""", 0)

no_branches_example_2 = ("""
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
<answer>4 * 5</answer>""", 0)


examples = [example_1, example_2, example_3, example_4, example_5, example_6, example_7, example_8, example_10, example_11, example_12, no_branches_example_1, no_branches_example_2]
short_examples = [example_1, example_7, example_8, example_10, example_11, example_12, no_branches_example_1]
double_examples = [(e1 + "\nExample:\n" + e2, max(m1, m2)) for e1, m1 in short_examples for e2, m2 in short_examples]
examples = examples + double_examples + examples
def get_format_and_examples(max_branching_points):
    filtered_examples = [example for example, branching_points in examples if branching_points <= max_branching_points]
    return multi_branch_format_prompt.format(max_branching_points=max_branching_points), filtered_examples