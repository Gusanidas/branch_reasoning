tri_branch_format_prompt= """
Place your reasoning between <think> and </think> tags.
Place your solution between <answer> and </answer> tags.
Dont write anything after </answer>.

You can branch your reasoning. Describe briefly three approaches that can lead to the solution.
All three approaches will be explored.
One branch between <a> and </a>, one branch between <b> and </b>, one branch between <c> and </c>.
Choose one of them by writing #a#, #b#, or #c#. Describe all three branches before choosing.
After branching, continue the reasoning of the branch you chose.
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
<c>
I will try adding all numbers first: 3 + 5 + 7 + 2 = 17, and see where that leads.
</c>
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
I will try multiplying 7 and 8, that is 7 * 8 = 56.
</a>
<b>
I will try multiplying 9 and 2, that is 9 * 2 = 18. And go from there.
</b>
<c>
I will try adding 7 and 8, that is 7 + 8 = 15. Then possibly add more.
</c>
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
I will try multiplying 20 and 5, that is 20 * 5 = 100. And go from there.
</b>
<c>
I will try multiplying 7 and 12, that is 7 * 12 = 84. And go from there.
</c>
#c#
With 7 * 12 = 84, I need 40 more to reach 124.
I have 2, 20, and 5 remaining.
Let me see what combinations I can make. There are a few possibilities here.
Let me branch out again.
<a>
I will try adding 2 * 20 = 40 to our current value of 84.
84 + 40 = 124. 
</a>
<b>
I will try subtracting 2 from our current value of 84.
84 - 2 = 82. 
</b>
<c>
I will try adding 5 * 2 = 10 and then adding 20 + 10 = 30.
84 + 30 = 114. This is still 10 short of our target.
</c>
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
Branching out.
<a>
7 * 12 = 84.
</a>
<b>
7 * 20 = 140.
</b>
<c>
20 * 5 = 100.
</c>
#a#
With 7 * 12 = 84, I need 40 more to reach 124.
I have 2, 20, and 5 remaining.
<a>
84 - 2 = 82.
</a>
<b>
84 + 40 = 124.
</b>
<c>
5 + 20 = 25.
84 + 25 = 109.
</c>
#b#
I have found the solution: 7 * 12 + 2 * 20 = 124.</think>
<answer>7 * 12 + 2 * 20</answer>""", 2)

example_5 = ("""
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
I will start writing three branches.
<a>
I will try multiplying 7 and 20, that is 7 * 20 = 140.
</a>
<b>
I will try multiplying 7 and 12, that is 7 * 12 = 84. And go from there.
</b>
<c>
I will try multiplying 12 and 9, that is 12 * 9 = 108. And go from there.
</c>
#b#
With 7 * 12 = 84, I need more to reach 169.
I have 2, 20, 5, and 9 remaining.
Let me branch out again.
<a>
84 - 2 = 82.
</a>
<b>
9 * 2 = 18.
84 + 18 = 102.
</b>
<c>
2 * 20 = 40.
84 + 40 = 124.
</c>
#c#
I have 7*12 + 2*20 = 124. And I have 5 and 9 remaining.
<a>
I will try multiplying 5 and 9, that is 5 * 9 = 45.
</a>
<b>
I will try adding 5 and 9, that is 5 + 9 = 14.
</b>
<c>
I will try subtracting 5 from 9, that is 9 - 5 = 4.
</c>
#a#
With 124, I can add 45, that is 124 + 45 = 169.
That is exactly our target!
</think>
I have found the solution: 7 * 12 + 2 * 20 + 9 * 5 = 169.
<answer>7 * 12 + 2 * 20 + 9 * 5</answer>""",3)


examples = [example_1, example_2, example_3, example_4, example_5]
def get_format_and_examples(max_branching_points):
    filtered_examples = [example for example, branching_points in examples if branching_points <= max_branching_points]
    return tri_branch_format_prompt.format(max_branching_points=max_branching_points), filtered_examples