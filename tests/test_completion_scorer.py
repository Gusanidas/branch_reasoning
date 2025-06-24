import unittest
import tempfile
import os
import json
import math
from collections import defaultdict
from branch_reasoning.scoring.completion_scorer import (
    CompletionScorer,
    _populate_avg_scores,
    _normalize_scores,
    _divide_branched_scores,
    _initialize_scores,
    _gather_metrics,
    _gather_branch_metrics,
    _save_highest_branch
)
from branch_reasoning.generation.completions import (
    Branch, BranchedCompletion, PromptCompletion, 
    ScoringData, Metadata
)


class TestCompletionScorer(unittest.TestCase):
    
    def setUp(self):
        self.wandb_logs = defaultdict(float)
        self.temp_file = None
        
    def tearDown(self):
        if self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def create_branch(self, completion, score=0, key="test_key"):
        branch = Branch(
            completion=completion,
            log_probs=None,
            ref_log_probs=None,
            score=score,
            key=key
        )
        branch.meta_scores = defaultdict(float)
        return branch
    
    def create_prompt_completion(self, prompt="Test prompt", branches_data_list=None):
        print("create_prompt_completion")
        print(f"branches_data_list: {branches_data_list}")
        if branches_data_list is None:
            branches_data_list = [[("completion1", 0.0), ("completion2", 0.0), ("completion3", 0.0), ("completion4", 0.0)]]
        
        branches_list = [[self.create_branch(completion=completion, score=score) for completion, score in branches_data] for branches_data in branches_data_list]
        branched_completion_list = [BranchedCompletion(branches=branches, score=0) for branches in branches_list]
        
        return PromptCompletion(
            prompt=prompt,
            scoring_data=ScoringData(nums=[10, 6], target=16),
            metadata=Metadata(solution="10 + 6", tag="test"),
            branched_completions=branched_completion_list,
        )
    
    def test_populate_avg_scores(self):
        branches_data_list = [[("comp1", 1.0), ("comp2", 2.0), ("comp3", 3.0)]]
        prompt_completion = self.create_prompt_completion(branches_data_list=branches_data_list)
        prompt_completion_list = [prompt_completion]
        
        _populate_avg_scores(prompt_completion_list)
        
        expected_avg = 2.0  # (1.0 + 2.0 + 3.0) / 3
        self.assertEqual(prompt_completion.branched_completions[0].score, expected_avg)
        
    def test_initialize_scores(self):
        prompt_completion = self.create_prompt_completion()
        prompt_completion_list = [prompt_completion]
        
        _initialize_scores(prompt_completion_list, init_value=5.0)
        
        for branch in prompt_completion.branched_completions[0].branches:
            self.assertEqual(branch.score, 5.0)
            self.assertIsInstance(branch.meta_scores, defaultdict)
            
    def test_normalize_scores_by_prompt(self):
        branches_list = [[("comp1", 1.0), ("comp2", 2.0)],
                         [("comp3", 3.0), ("comp4", 4.0)]]
        prompt_completion = self.create_prompt_completion(branches_data_list=branches_list)
        
        _normalize_scores([prompt_completion], normalize_by_prompt=True)
        
        # Check that scores have been normalized
        scores = [bc.score for bc in prompt_completion.branched_completions]
        mean = sum(scores) / len(scores)
        self.assertAlmostEqual(mean, 0.0, places=5)
        
    def test_divide_branched_scores(self):
        branches = [self.create_branch("comp1", 2.0), self.create_branch("comp2", 4.0)]
        branched_completion = BranchedCompletion(branches=branches, score=3.0)
        prompt_completion = PromptCompletion(
            prompt="test",
            scoring_data=ScoringData(nums=[1, 2], target=3),
            metadata=Metadata(solution="1+2", tag="test"),
            branched_completions=[branched_completion]
        )
        
        _divide_branched_scores([prompt_completion])
        
        # Total branch scores = 6.0, branched_completion.score = 3.0
        # ratio = 6.0/3.0 = 2.0
        # Each branch score should be divided by (ratio * num_branches) = 2.0 * 2 = 4.0
        self.assertAlmostEqual(branches[0].score, 0.5, places=5)  # 2.0 / 4.0
        self.assertAlmostEqual(branches[1].score, 1.0, places=5)  # 4.0 / 4.0
        
    def test_gather_metrics(self):
        prompt_completion = self.create_prompt_completion()
        prompt_completion.branched_completions[0].score = 2.5
        
        _gather_metrics([prompt_completion], self.wandb_logs, mark="test")
        
        self.assertGreater(self.wandb_logs["post_score_test/avg_score"], 0)
        self.assertEqual(self.wandb_logs["post_score_test/avg_score_steps"], 1)
        
    def test_gather_branch_metrics(self):
        branches = [
            self.create_branch("comp1", 1.0, key="base_0_0_0_0"),
            self.create_branch("comp2", 2.0, key="base_0_1_0_0")
        ]
        branched_completion = BranchedCompletion(branches=branches, score=None)
        prompt_completion = PromptCompletion(
            prompt="test",
            scoring_data=ScoringData(nums=[1, 2], target=3),
            metadata=Metadata(solution="1+2", tag="test"),
            branched_completions=[branched_completion]
        )
        
        _gather_branch_metrics([prompt_completion], self.wandb_logs)
        
        self.assertEqual(self.wandb_logs["branch_scores/avg_base_score"], 1.0)
        self.assertEqual(self.wandb_logs["branch_scores/avg_branched_score"], 2.0)
        
    def test_save_highest_branch(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl').name
        
        branches = [
            self.create_branch("low score", 1.0),
            self.create_branch("high score", 5.0),
            self.create_branch("medium score", 3.0)
        ]
        for branch in branches:
            branch.meta_scores = {"test_metric": 10.0, "test_metric_steps": 2}
            
        branched_completion = BranchedCompletion(branches=branches, score=None)
        prompt_completion = PromptCompletion(
            prompt="test prompt",
            scoring_data=ScoringData(nums=[1, 2], target=3),
            metadata=Metadata(solution="1+2", tag="test"),
            branched_completions=[branched_completion]
        )
        
        _save_highest_branch([prompt_completion], output_file=self.temp_file)
        
        # Read and verify the saved entry
        with open(self.temp_file, 'r') as f:
            entry = json.loads(f.readline())
            
        self.assertEqual(entry["prompt"], "test prompt")
        self.assertEqual(entry["completion"], "high score")
        self.assertEqual(entry["score"], 5.0)
        self.assertEqual(entry["test_metric"], 10.0)
        self.assertEqual(entry["test_metric_steps"], 2)
        
    def test_completion_scorer_initialization(self):
        def dummy_scorer(branched_completion, prompt, **kwargs):
            return 1.0
            
        scorer = CompletionScorer([dummy_scorer], normalize_by_prompt=True)
        self.assertEqual(len(scorer.scoring_functions), 1)
        self.assertTrue(scorer.normalize_by_prompt)
        
    def test_completion_scorer_score_completions(self):
        wandb_logs = defaultdict(float)
        def simple_scorer(branched_completion, prompt, **kwargs):
            for branch in branched_completion.branches:
                branch.score += 1.0
                branch.meta_scores["simple_score"] += 1.0
            return 1.0
            
        scorer = CompletionScorer([simple_scorer], normalize_by_prompt=False)
        branches_list = [[("comp1", 1.0), ("comp2", 2.0)],
                         [("comp3", 3.0), ("comp4", 4.0)]]
        prompt_completion_list = [self.create_prompt_completion(branches_data_list=branches_list)]
        
        prompt_completion_list = scorer.score_completions(prompt_completion_list, wandb_logs)
        
        self.assertEqual(wandb_logs["score/simple_scorer"], 2.0)
        self.assertEqual(wandb_logs["score/simple_scorer_steps"], 2.0)
            
    def test_multiple_scoring_functions(self):
        def scorer1(branched_completion, prompt, **kwargs):
            for branch in branched_completion.branches:
                branch.score += 1.0
            return 1.0
            
        def scorer2(branched_completion, prompt, **kwargs):
            for branch in branched_completion.branches:
                branch.score += 2.0
            return 2.0
            
        scorer = CompletionScorer([scorer1, scorer2], normalize_by_prompt=False)
        prompt_completion = self.create_prompt_completion()
        
        # Reset initial scores to 0
        for branch in prompt_completion.branched_completions[0].branches:
            branch.score = 0
            
        result = scorer.score_completions([prompt_completion], self.wandb_logs)
        
        # Check that both scoring functions were applied
        self.assertEqual(self.wandb_logs["score/scorer1"], 1.0)
        self.assertEqual(self.wandb_logs["score/scorer2"], 2.0)
        self.assertEqual(self.wandb_logs["score/total_score"], 3.0)


if __name__ == "__main__":
    unittest.main()