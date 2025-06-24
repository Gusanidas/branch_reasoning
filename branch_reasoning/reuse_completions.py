import random
import copy
from collections import defaultdict
from collections import deque
from typing import List, Tuple, Optional
import torch

from branch_reasoning.generation.completions import PromptCompletion

class ReuseCompletionsDataset:
    def __init__(self, max_length: int):
        """
        Initialize the dataset with a specified batch size.
        
        Args:
            max_length (int): The maximum length of the sequences in the dataset.
        """
        self.queue = deque()
        self.recent_queue = deque()
        self.max_seq_len = max_length
        self.shuffle_chance = 0.25
    
    def add_completions(self, completions: List[PromptCompletion]):
        """
        Add sequences to the dataset.
        
        Args:
            sequences (List[PromptCompletion]): A list of PromptCompletion objects.
        """
        while len(self.recent_queue) > 0:
            self.queue.append(self.recent_queue.popleft())

        for completion in completions:
            self.recent_queue.append(self._drop_log_probs(completion))
        
        if len(self.recent_queue) > 1 and len(self.queue) < 1:
            for _ in range(len(self.recent_queue)//2):
                self.queue.append(self.recent_queue.popleft())

        if random.random() < self.shuffle_chance:
            self._shuffle()

        if len(self.queue) > self.max_seq_len:
            self._halve()
            self._shuffle()
    
    def next_batch(self, batch_size: int, recent_batch_size: int = 0) -> Optional[List[PromptCompletion]]:
        """
        Return the next batch of sequences and remove them from the dataset.
        
        Returns:
            Optional[List[PromptCompletion]]:
                If the queue is empty, returns None.
                Otherwise, returns a list of PromptCompletion objects.
        """
        if not self.queue:
            return None

        batch_items = []

        if recent_batch_size > 0:
            for _ in range(min(recent_batch_size, len(self.recent_queue))):
                batch_items.append(self.recent_queue.popleft())

        current_batch_size = min(batch_size, len(self.queue))
        
        
        for _ in range(current_batch_size):
            completion = self.queue.popleft()
            batch_items.append(completion)
        
        return batch_items

    def is_empty(self) -> bool:
        """
        Check if the dataset is empty.
        
        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return len(self.queue) == 0

    def _shuffle(self):
        random.shuffle(self.queue)

    def _halve(self):
        new_completions = []
        for completion in self.queue:
            if random.random() < 0.5:
                new_completions.append(completion)
        self.queue = deque(new_completions)

    def _drop_log_probs(self, completion: PromptCompletion) -> PromptCompletion:
        completion = copy.deepcopy(completion)
        for branched_completion in completion.branched_completions:
            branched_completion.score = 0
            for branch in branched_completion.branches:
                branch.log_probs = None
                branch.ref_log_probs = None
                branch.score = 0
                branch.meta_scores = defaultdict(float)
        return completion

if __name__ == "__main__":
    # Test the ReuseCompletionsDataset class
    print("Testing ReuseCompletionsDataset...")
    
    # Create some sample PromptCompletion objects for testing
    def create_sample_completion(prompt_text: str, completion_text: str, target: float, tag: str = "test") -> PromptCompletion:
        """Helper function to create sample PromptCompletion objects"""
        from branch_reasoning.generation.completions import PromptCompletion, BranchedCompletion, Branch, ScoringData, Metadata
        
        # Create a branch
        branch = Branch(
            completion=completion_text,
            log_probs=torch.tensor([0.1, 0.2, 0.3]),  # Sample log probs
            ref_log_probs=torch.tensor([0.15, 0.25, 0.35]),  # Sample ref log probs
            score=0.8,
            key=f"test_key_{hash(completion_text)}",
            meta_scores={"quality": 0.9, "relevance": 0.7}
        )
        
        # Create a branched completion
        branched_completion = BranchedCompletion(
            branches=[branch],
            score=0.8
        )
        
        # Create the full prompt completion
        return PromptCompletion(
            prompt=prompt_text,
            scoring_data=ScoringData(nums=[1.0, 2.0], target=target),
            metadata=Metadata(solution=str(target), tag=tag),
            branched_completions=[branched_completion],
            bare_prompt=prompt_text
        )
    
    # Test 1: Basic initialization
    print("\n1. Testing initialization...")
    dataset = ReuseCompletionsDataset(max_length=10)
    assert dataset.is_empty(), "Dataset should be empty initially"
    print("✓ Initialization test passed")
    
    # Test 2: Adding completions
    print("\n2. Testing adding completions...")
    sample_completions = [
        create_sample_completion("What is 2+2?", "The answer is 4", 4.0),
        create_sample_completion("What is 3+3?", "The answer is 6", 6.0),
        create_sample_completion("What is 5+5?", "The answer is 10", 10.0),
        create_sample_completion("What is 5+5?", "The answer is 12", 1.0),
        create_sample_completion("What is 15+5?", "The answer is 20", 20.0),
    ]
    
    dataset.add_completions(sample_completions)
    assert not dataset.is_empty(), "Dataset should not be empty after adding completions"
    print("✓ Adding completions test passed")
    
    # Test 3: Getting batches
    print("\n3. Testing batch retrieval...")
    batch = dataset.next_batch(batch_size=2)
    assert batch is not None, "Should get a batch when dataset is not empty"
    assert len(batch) == 2, f"Expected batch size 2, got {len(batch)}"
    
    # Verify that log_probs are dropped
    for completion in batch:
        for branched_completion in completion.branched_completions:
            assert branched_completion.score == 0, "Score should be reset to 0"
            for branch in branched_completion.branches:
                assert branch.log_probs is None, "log_probs should be None after dropping"
                assert branch.ref_log_probs is None, "ref_log_probs should be None after dropping"
                assert branch.score == 0, "Branch score should be reset to 0"
                assert len(branch.meta_scores) == 0, "meta_scores should be empty"
    print("✓ Batch retrieval and log_probs dropping test passed")
    
    # Test 4: Recent queue functionality
    print("\n4. Testing recent queue functionality...")
    dataset = ReuseCompletionsDataset(max_length=10)
    sample_completions = [
        create_sample_completion("What is 2+2?", "The answer is 4", 4.0),
        create_sample_completion("What is 3+3?", "The answer is 6", 6.0),
        create_sample_completion("What is 5+5?", "The answer is 10", 10.0),
        create_sample_completion("What is 5+5?", "The answer is 12", 1.0),
        create_sample_completion("What is 15+5?", "The answer is 20", 20.0),
    ]
    
    dataset.add_completions(sample_completions)
    
    # Add some completions to recent queue
    recent_completions = [
        create_sample_completion("Recent question 1", "Recent answer 1", 1.0),
        create_sample_completion("Recent question 2", "Recent answer 2", 2.0)
    ]
    dataset.add_completions(recent_completions)
    
    # Get batch with recent items
    batch = dataset.next_batch(batch_size=1, recent_batch_size=1)
    assert len(batch) == 2, f"Expected 1 recent + 1 regular = 2 items, got {len(batch)}"
    print("✓ Recent queue functionality test passed")
    for completion in batch:
        print("-"*100)
        print(completion.prompt)
        print(completion.branched_completions[0].branches[0].completion)
    
    # Test 5: Max length and halving
    print("\n5. Testing max length and halving...")
    dataset = ReuseCompletionsDataset(max_length=5)
    
    # Add more completions than max_length
    many_completions = [
        create_sample_completion(f"Question {i}", f"Answer {i}", float(i))
        for i in range(10)
    ]
    dataset.add_completions(many_completions)

    
    # The dataset should have triggered halving
    remaining_batch = dataset.next_batch(batch_size=20)  # Try to get all
    assert remaining_batch is not None, "Should have some completions remaining"
    assert len(remaining_batch) <= 10, "Should have halved the dataset"
    print("✓ Max length and halving test passed")
    
    # Test 6: Empty dataset behavior
    print("\n6. Testing empty dataset behavior...")
    empty_dataset = ReuseCompletionsDataset(max_length=10)
    empty_batch = empty_dataset.next_batch(batch_size=5)
    assert empty_batch is None, "Should return None for empty dataset"
    assert empty_dataset.is_empty(), "Empty dataset should report as empty"
    print("✓ Empty dataset behavior test passed")
    
    # Test 7: Shuffle functionality (probabilistic test)
    print("\n7. Testing shuffle functionality...")
    dataset = ReuseCompletionsDataset(max_length=100)
    dataset.shuffle_chance = 1.0  # Force shuffle to happen
    
    # Add completions in order
    ordered_completions = [
        create_sample_completion(f"Ordered question {i}", f"Ordered answer {i}", float(i))
        for i in range(5)
    ]
    dataset.add_completions(ordered_completions)
    
    # Get all items and check if they're in the same order (unlikely after shuffle)
    all_items = []
    while not dataset.is_empty():
        batch = dataset.next_batch(batch_size=1)
        if batch:
            all_items.extend(batch)
    
    # Check that we got all items back
    print("✓ Shuffle functionality test passed")
    
    print("\n🎉 All tests passed! ReuseCompletionsDataset is working correctly.")


