import re
from typing import Optional, List, Dict, Tuple, Union
import torch
from collections import deque



def create_keyword_regex(branching_factor=2):
    if not (2 <= branching_factor <= 9):
        raise ValueError("Branching factor must be between 2 and 9 inclusive")
    
    if not hasattr(create_keyword_regex, '_pattern_cache'):
        create_keyword_regex._pattern_cache = {}
    
    if branching_factor in create_keyword_regex._pattern_cache:
        return create_keyword_regex._pattern_cache[branching_factor]
    
    keywords = [f"#{chr(97 + i)}#" for i in range(branching_factor)]
    pattern_str = r"(?:" + "|".join(keywords) + r")"
    compiled_pattern = re.compile(pattern_str, flags=re.DOTALL)
    create_keyword_regex._pattern_cache[branching_factor] = compiled_pattern
    
    return compiled_pattern


def find_subarray_nth_occurrence(text: str, substring: str, n: int) -> int:
    """
    Find the nth occurrence of a substring within a string.
    
    Args:
        text: The text to search within.
        substring: The substring to find.
        n: The occurrence number (1-based).
        
    Returns:
        The index of the nth occurrence or -1 if not found.
    """
    start = 0
    for _ in range(n):
        index = text.find(substring, start)
        if index == -1:
            return -1
        start = index + 1
    return index

def get_new_branches_for_point(text: str, branch_number: int, branching_factor: int = 2) -> Optional[str]:
    """
    Find the nth occurrence of keyword_regex after KEYWORD,
    then check if the surrounding text matches branch_regex.
    If it does, swap the ending between #a# and #b#.
    
    Args:
        text: The input text.
        branch_number: Which keyword occurrence to modify (1-based).
        
    Returns:
        Modified text or None if branch not found.
    """
        
    keyword_regex = create_keyword_regex(branching_factor)

    search_start_pos = 0
    keyword_matches = list(keyword_regex.finditer(text, pos=search_start_pos))
    if branch_number <= 0 or branch_number > len(keyword_matches):
        return []
    
    keyword_match = keyword_matches[branch_number - 1]
    if branch_number>1:
        last_keyword_match = keyword_matches[branch_number - 2]
        last_end = last_keyword_match.end()
    else:
        last_end = 0
    keyword_start, keyword_end = keyword_match.start(), keyword_match.end()
    possible_branch = text[last_end:keyword_end]
    think_end = text.find("</think>", search_start_pos)
    think_start = text.find("<think>", search_start_pos)
    if think_end < keyword_start or think_start > keyword_end:
        return []
    prefix = text[:keyword_start]
    keywords = [f"#{chr(97 + i)}#" for i in range(branching_factor)]
    for keyword in keywords:
        r = []
        if possible_branch.endswith(keyword):
            for new_keyword in keywords:
                if new_keyword != keyword:
                    r.append(prefix + new_keyword)
            return r
    return []

def get_new_branches(all_completions: Dict[str, Tuple[str, str]], branching_factor: int = 2, max_branching_points: int = 3) -> Dict[str, Tuple[str, str]]:
    """
    Process the dictionary of completions and generate new branches.
    
    Args:
        all_completions: Dictionary with keys like "base_0_0_0" and string values.
        all_mask: Dictionary with base keys and corresponding mask strings.
        
    Returns:
        Dictionary containing new keys and modified completions.
    """
    new_branches = {}
    
    for key, (prompt, completion) in all_completions.items():
        parts = key.split("_")
        if len(parts) < 3:
            continue
            
        base_key = parts[0]
        iter_num = parts[1]
        branches = parts[2:]
        
        for i in range(min(max_branching_points, len(branches)) - 1, -1, -1):
            if branches[i] == "0":
                branches_copy = branches.copy()
                new_key = f"{base_key}_{iter_num}_{'_'.join(branches_copy)}"
                
                new_branches_for_point = get_new_branches_for_point(completion, i + 1, branching_factor)
                
                for j, new_branch in enumerate(new_branches_for_point):
                    branches_copy[i] = str(j+1)
                    new_key = f"{base_key}_{iter_num}_{'_'.join(branches_copy)}"
                    new_branches[new_key] = (prompt, new_branch)
            else:
                # Skip if branch is already ">0"
                break
                
    return new_branches

class QueueDataset:
    def __init__(self, tokenizer, batch_size: int, max_seq_len: int = 1500):
        """
        Initialize the dataset with a specified batch size.
        
        Args:
            batch_size (int): The number of items to include in each batch.
        """
        self.batch_size = batch_size
        self.queue = deque()  # Queue to store (key, tensor) pairs
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
    
    def add_sequences(self, sequences: List[Tuple[str, Union[str, Tuple[str, str]]]]):
        """
        Add sequences to the dataset.
        
        Args:
            sequences (List[Tuple[str, Union[str, Tuple[str, str]]]]): A list of (key, value) pairs.
                The key is a string identifier for the sequence.
                The value can be either a string (completion) or a tuple (prompt, completion).
        """
        for key, value in sequences:
            # Handle both string and tuple formats
            if isinstance(value, tuple):
                prompt, completion = value
            else:
                completion = value
                prompt = None  # For backward compatibility
            
            tokens = self.tokenizer.encode(completion, add_special_tokens=False)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len-3] + tokens[-3:]
            
            # Store the full value (either string or tuple)
            self.queue.append((key, value))
    
    def next_batch(self) -> Optional[Tuple[List[str], torch.Tensor, torch.Tensor]]:
        """
        Return the next batch of sequences and remove them from the dataset.
        
        Returns:
            Optional[Tuple[List[str], torch.Tensor, torch.Tensor]]:
                If the queue is empty, returns None.
                Otherwise, returns a tuple containing:
                - List[str]: A list of keys for the sequences in the batch.
                - torch.Tensor: A tensor of shape (batch_size, seq_len) containing the sequences,
                  aligned at the last token.
                - torch.Tensor: A tensor of shape (batch_size, seq_len) containing the attention mask,
                  where 1 indicates tokens that should be attended to and 0 indicates tokens that should be ignored.
        """
        if not self.queue:
            return None
        
        current_batch_size = min(self.batch_size, len(self.queue))
        
        batch_items = []
        batch_keys = []
        
        for _ in range(current_batch_size):
            key, tensor = self.queue.popleft()
            batch_items.append(tensor)
            batch_keys.append(key)
        
        return batch_keys, batch_items

    def is_empty(self) -> bool:
        """
        Check if the dataset is empty.
        
        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return len(self.queue) == 0

if __name__ == "__main__":
    print("\nTesting get_new_branches:")
    print("-" * 20)

    
    sample_completion_str1 = "My Assistant:<think>  the new branches.<a> I am </a> But I am not\n<b>other\n</b>\n#a#\n</b>\n#b#\n</b>\n#a#\n</b>\n#b#\n</b>\n#a# </think> <answer> I am not </answer>"
    sample_completion_str2 = "My Assistant:<think> t\n<b>other\n</b>\n#b# </think> <answer> I am not </answer>"
    completion_strs = [sample_completion_str1]
    completions = completion_strs
    all_completions = {}
    all_masks = {}
    for i, completion in enumerate(completions):
        base_key = "base"
        key = f"{base_key}_{i}_0_0_0_0_0"
        all_completions[key] = completion

    for sample_completion in completions:
        print(f"sample completion: {sample_completion}")
        print("0-0-0-0-0-0-0-0-0")

    # Test the function
    new_branches = get_new_branches(all_completions, branching_factor=3, max_branching_points=2)
    print(f"len new_branches: {len(new_branches)}")
    
    print("-=--=--=-=-=-=-=-=-=-=-=-=-")
    print("New branches found:")
    for key, branch in new_branches.items():
        print(f"Key: {key}")
        print(f"Branch: {branch}")
        print("-" * 10)