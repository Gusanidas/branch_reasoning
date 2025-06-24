import unittest
from collections import defaultdict
from branch_reasoning.scoring.scoring_functions import (
    match_format_exactly,
    match_format_loosely,
    match_format_approximately,
    rate_countdown_answer,
    get_branch_regex,
    get_loose_branch_regex,
    extract_thinking,
    score_branch_format,
    score_branch_format_loose,
    score_branch_format_approx,
    _rate_countdown_answer_strict,
    _match_branch_fromat,
    _match_loose_branch_fromat,
    _score_approx
)
from branch_reasoning.generation.completions import Branch, BranchedCompletion, ScoringData


class TestScoringFunctions(unittest.TestCase):
    
    def setUp(self):
        self.prompt = "Test prompt"
        self.wandb_logs = defaultdict(float)
        
    def create_branch(self, completion, score=0):
        branch = Branch(
            completion=completion,
            log_probs=None,
            ref_log_probs=None,
            score=score,
            key="test_key"
        )
        branch.meta_scores = defaultdict(float)
        return branch
    
    def test_match_format_exactly(self):
        completion = "<think>This is thinking</think>\n<answer>42</answer>"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        
        score = match_format_exactly(branched_completion, self.prompt)
        self.assertEqual(score, 2.0)
        self.assertEqual(branch.score, 2.0)
        self.assertEqual(branch.meta_scores["match_format_exactly"], 2.0)
        
    def test_match_format_exactly_invalid(self):
        completion = "No proper format here"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        
        score = match_format_exactly(branched_completion, self.prompt)
        self.assertEqual(score, 0.0)
        self.assertEqual(branch.score, 0.0)
        
    def test_match_format_loosely(self):
        completion = "Some text <think>thinking</think> more text <answer>42</answer> end"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        
        score = match_format_loosely(branched_completion, self.prompt)
        self.assertEqual(score, 1.0)
        self.assertEqual(branch.score, 1.0)
        self.assertEqual(branch.meta_scores["match_format_loosely"], 1.0)
        
    def test_match_format_approximately(self):
        completion = "<think>This is thinking</think><answer>42</answer>"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        
        score = match_format_approximately(branched_completion, self.prompt)
        self.assertGreater(score, 0)
        self.assertGreater(branch.score, 0)
        
    def test_rate_countdown_answer_strict(self):
        completion = "<answer>10 + 6</answer>"
        nums = [10, 6]
        target = 16
        
        score = _rate_countdown_answer_strict(completion, nums, target)
        self.assertEqual(score, 5.0)
        
    def test_rate_countdown_answer_strict_wrong(self):
        completion = "<answer>10 * 6</answer>"
        nums = [10, 6]
        target = 16
        
        score = _rate_countdown_answer_strict(completion, nums, target)
        self.assertEqual(score, 0.0)
        
    def test_rate_countdown_answer_strict_invalid_nums(self):
        completion = "<answer>10 + 7</answer>"
        nums = [10, 6]
        target = 17
        
        score = _rate_countdown_answer_strict(completion, nums, target)
        self.assertEqual(score, 0.0)
        
    def test_rate_countdown_answer(self):
        completion = "<answer>10 + 6</answer>"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        scoring_data = ScoringData(nums=[10, 6], target=16)
        
        score = rate_countdown_answer(
            branched_completion, 
            self.prompt, 
            scoring_data,
            mode="eq",
            wandb_logs=self.wandb_logs
        )
        self.assertEqual(score, 5.0)
        self.assertEqual(branch.score, 5.0)
        
    def test_get_branch_regex(self):
        regex2 = get_branch_regex(2)
        text = "<a>content a</a><b>content b</b>#a#"
        self.assertIsNotNone(regex2.search(text))
        
        regex3 = get_branch_regex(3)
        text = "<a>content a</a><b>content b</b><c>content c</c>#b#"
        self.assertIsNotNone(regex3.search(text))
        
        with self.assertRaises(ValueError):
            get_branch_regex(1)
            
        with self.assertRaises(ValueError):
            get_branch_regex(5)
            
    def test_get_loose_branch_regex(self):
        regex2 = get_loose_branch_regex(2)
        text = "<a>content a</a><b>content b</b>#a#"
        self.assertIsNotNone(regex2.search(text))
        
    def test_extract_thinking(self):
        text = "Before <think>This is the thinking part</think> After"
        result = extract_thinking(text)
        self.assertEqual(result, "This is the thinking part")
        
        text_no_think = "No thinking tags here"
        result = extract_thinking(text_no_think)
        self.assertEqual(result, "")
        
    def test_score_branch_format(self):
        completion = "<think><a>option a</a><b>option b</b>#a#</think><answer>42</answer>"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        
        score = score_branch_format(branched_completion, 2, 2, self.prompt)
        self.assertGreater(score, 0)
        self.assertGreater(branch.score, 0)
        
    def test_score_branch_format_loose(self):
        completion = "<think><a>option a</a><b>option b</b>#a#</think><answer>42</answer>"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        
        score = score_branch_format_loose(branched_completion, 2, 2, self.prompt)
        self.assertGreater(score, 0)
        self.assertGreater(branch.score, 0)
        
    def test_score_branch_format_approx(self):
        completion = "<think><a>option a</a><b>option b</b>#a#</think><answer>42</answer>"
        branch = self.create_branch(completion)
        branched_completion = BranchedCompletion(branches=[branch], score=None)
        
        score = score_branch_format_approx(branched_completion, 2, 2, self.prompt)
        self.assertGreater(score, 0)
        self.assertGreater(branch.score, 0)
        
    def test_match_branch_format(self):
        text = "<a>content a</a><b>content b</b>#a#"
        regex = get_branch_regex(2)
        score = _match_branch_fromat(text, 2, regex)
        self.assertGreater(score, 0)
        
    def test_match_loose_branch_format(self):
        text = "<a>content a</a><b>content b</b>#a#"
        regex = get_loose_branch_regex(2)
        score = _match_loose_branch_fromat(text, 2, regex)
        self.assertGreater(score, 0)
        
    def test_score_approx(self):
        text = "<a>content</a><b>content</b>#a#"
        branch_keywords = ["<a>", "<b>", "</a>", "</b>"]
        branch_choice_keywords = ["#a#", "#b#"]
        score = _score_approx(text, branch_keywords, branch_choice_keywords, 2)
        self.assertGreaterEqual(score, 0)


if __name__ == "__main__":
    unittest.main()