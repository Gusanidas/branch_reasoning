quad_branch_format_prompt= """
Place your reasoning between <think> and </think> tags.
Place your solution between <answer> and </answer> tags.
Dont write anything after </answer>.

You can branch your reasoning. Describe briefly four different approaches.
Choose one of them by writing #a#, #b#, #c#, or #d#. Describe all four branches before choosing.
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
<d>
I will try multiplying 3 and 2, that is 3 * 2 = 6, and then combine with the other numbers.
</d>
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
<d>
I will try multiplying 8 and 2, that is 8 * 2 = 16, and then add 9 and 7.
</d>
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
Numbers: 7, 12, 2, 20, 5, 1
Target: 125
<think>
Some starting options.
<a>
7*20 = 140.
</a>
<b>
7*12 = 84.
</b>
<c>
20*5 = 100.
</c>
<d>
12*2 = 24.
</d>
#b#
84, I need 40 more to reach 124.
2, 20 and 5 are remaining.
<a>
2*20 = 40.
84 + 40 = 124. 
</a>
<b>
84 - 2 = 82. 
</b>
<c>
5*2 = 10. 10 + 20 = 30.
84 + 30 = 114.
</c>
<d>
20 + 5 = 25.
84 + 25 = 109.
</d>
#a#
124 + 1 = 125.
</think>
I have found the solution: 7 * 12 + 2 * 20 + 1 = 125.
<answer>7 * 12 + 2 * 20 + 1</answer>
""", 2)

example_4 = ("""
Numbers: 9, 8, 7, 2
Target: 33
<think>
I will write different branches.
<a>
I will try multiplying 7 and 8, that is 7 * 8 = 56.
</a>
<b>
I will try adding 7 and 8, that is 7 + 8 = 15. Then possibly add more.
</b>
<c>
I will try multiplying 9 and 2, that is 9 * 2 = 18. And go from there.
</c>
<d>
I will try multiplying 8 and 2, that is 8 * 2 = 16, and then add 9 and 7.
</d>
#c#
With 9 * 2 = 18, I need to add something to reach 33.
I need to add 15 more.
I have 7 and 8 remaining. If I add them: 7 + 8 = 15.
So, 18 + 15 = 33.
That is exactly our target!
</think>
I have found the solution: 9 * 2 + 7 + 8 = 33.
<answer>9 * 2 + 7 + 8</answer>
""", 1)

example_5 = ("""
Numbers: 7, 12, 2, 20, 5, 1
Target: 113
<think>
Branching out.
<a>
7*20 = 140.
</a>
<b>
7*12 = 84.
</b>
<c>
20*5 = 100.
</c>
<d>
12*2 = 24.
</d>
#b#
84, I need 40 more to reach 124.
2, 20 and 5 are remaining.
<a>
2*20 = 40.
84 + 40 = 124. 
</a>
<b>
84 - 2 = 82. 
</b>
<c>
5*2 = 10. 10 + 20 = 30.
84 + 30 = 114.
</c>
<d>
20 + 5 = 25.
84 + 25 = 109.
</d>
#c#
114 - 1 = 113.
</think>
I have found the solution: 7*12 + 20 + 2*5 -1 = 113
<answer>7*12 + 20 + 2*5 -1</answer>
""", 2)


example_6 = ("""
Numbers: 7, 12, 2, 20, 5, 9
Target: 169
<think>
I will start writing four branches.
<a>
7*20 = 140.
</a>
<b>
7*12 = 84.
</b>
<c>
12*9 = 108.
</c>
<d>
20*5 = 100.
</d>
#b#
With 7 * 12 = 84, I need more to reach 169.
I have 2, 20, 5, and 9 remaining.
Let me branch out again.
<a>
84 - 2 = 82.
</a>
<b>
84 + 40 = 124.
</b>
<c>
9*2 = 18.
84 + 18 = 102.
</c>
<d>
20 + 5 = 25.
84 + 25 * 2 = 84 + 50 = 134.
</d>
#b#
I have 7*12 + 2*20 = 124. And I have 5 and 9 remaining.
Let me branch out again.
<a>
5 * 9 = 45.
</a>
<b>
5 + 9 = 14.
</b>
<c>
I will try subtracting 5 from 9, that is 9 - 5 = 4.
</c>
<d>
I will try 9^2 = 81, but that would be using 9 twice, which isn't allowed.
Let me try 9 / 5 = 1.8, which isn't an integer, so this isn't helpful.
</d>
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
There are many options open, I will branch out.
<a>
I will try multiplying 3 and 2, that is 3 * 2 = 6, and then combine with the other numbers.
</a>
<b>
I will multiply 7 and 3, that is 7 * 3 = 21. And go from there.
</b>
<c>
I will try adding all numbers first: 3 + 5 + 7 + 2 = 17, and see where that leads.
</c>
<d>
I will try to multiply 5 and 7, that is 5 * 7 = 35. And go from there.
</d>
#d#
With 35, I can multiply by 2, that is 35 * 2 = 70.
Then I can subtract 3, that is 70 - 3 = 67.
That is the solution.
</think>

I have found the solution: 7*5*2 - 3 = 67.
<answer>7*5*2 - 3</answer>
""", 1)

example_10 = ("""
Numbers: 23,49, 5, 85, 26
Target: 41
<think>
<a>85- 49</a> <b>23 + 5</b><c> 23 + 49</c><d>85 - 26</d>
#c#

23 + 49 = 72
<a>85-72 = 13</a> <b>72 - 26 = 46</b><c>72-5 = 67</c><d>72 + 5 = 77</d>
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
<a>26 + 49 = 75</a> <b>26 - 5 = 21</b><c>26 + 5 = 31</c><d>26 - 23 = 3</d>
#d#

26-23 = 3
85 - 3 = 82.

</think>
I have found the solution: 85 + 26 - 23 = 82.
<answer>85 + 26 - 23</answer>
""",1)

example_12 = ("""
Numbers: 20, 44, 50, 10, 12, 18
Target: 92
<think>
I will branch early and explore each path separately.
<a>20 + 10</a> <b>20 + 50</b><c>20 + 44</c><d>20 + 12</d>
#c#

20 + 44 = 64
<a>64 + 10 = 74</a> <b>64 + 12 = 76</b><c>64 + 18 = 82</c><d>64 -50 + 10 = 24</d>
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


examples = [example_1, example_2, example_3, example_4, example_5, example_6, example_7]
def get_format_and_examples(max_branching_points):
    filtered_examples = [example for example, branching_points in examples if branching_points <= max_branching_points]
    return quad_branch_format_prompt.format(max_branching_points=max_branching_points), filtered_examples