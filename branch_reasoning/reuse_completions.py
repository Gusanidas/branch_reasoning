import random

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
        self.max_seq_len = max_length
    
    def add_completions(self, completions: List[PromptCompletion]):
        """
        Add sequences to the dataset.
        
        Args:
            sequences (List[PromptCompletion]): A list of PromptCompletion objects.
        """
        for completion in completions:
            self.queue.append(self._drop_log_probs(completion))

        if len(self.queue) > self.max_seq_len:
            self._halve()
            self._shuffle()
    
    def next_batch(self, batch_size: int) -> Optional[List[PromptCompletion]]:
        """
        Return the next batch of sequences and remove them from the dataset.
        
        Returns:
            Optional[List[PromptCompletion]]:
                If the queue is empty, returns None.
                Otherwise, returns a list of PromptCompletion objects.
        """
        if not self.queue:
            return None
        
        current_batch_size = min(batch_size, len(self.queue))
        
        batch_items = []
        
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
        for branched_completion in completion.branched_completions:
            for branch in branched_completion.branches:
                branch.log_probs = None
                branch.ref_log_probs = None
        return completion
