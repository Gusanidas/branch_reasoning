import pytest
from branch_reasoning.generation.completions import (
    _pack_into_return_datatypes,
    PromptCompletion,
    BranchedCompletion,
    Branch,
    ScoringData,
    Metadata
)


def test_pack_into_return_datatypes_single_prompt_single_completion():
    """Test packing a single prompt with a single completion."""
    all_completions = {"1#0_0_0_0_0": "The answer is 4."}
    all_prompts = {"1#0": "What is 2 + 2?"}
    all_numbers = {"1#0": [2, 2]}
    all_targets = {"1#0": 4}
    all_tags = {"1#0": "math"}
    all_solutions = {"1#0": "4"}
    
    result = _pack_into_return_datatypes(
        all_completions, all_prompts, all_numbers, 
        all_targets, all_tags, all_solutions
    )
    
    assert len(result) == 1
    prompt_completion = result[0]
    assert prompt_completion.prompt == "What is 2 + 2?"
    assert prompt_completion.scoring_data.nums == [2, 2]
    assert prompt_completion.scoring_data.target == 4
    assert prompt_completion.metadata.solution == "4"
    assert prompt_completion.metadata.tag == "math"
    assert len(prompt_completion.branched_completions) == 1
    assert len(prompt_completion.branched_completions[0].branches) == 1
    assert prompt_completion.branched_completions[0].branches[0].completion == "The answer is 4."


def test_pack_into_return_datatypes_multiple_completions():
    """Test packing multiple completions for the same prompt."""
    all_completions = {
        "1#0_0_0_0_0": "First completion",
        "1#0_1_0_0_0": "Second completion",
        "1#0_2_0_0_0": "Third completion"
    }
    all_prompts = {"1#0": "Test prompt"}
    all_numbers = {"1#0": [1, 2, 3]}
    all_targets = {"1#0": 6}
    all_tags = {"1#0": "test"}
    all_solutions = {"1#0": "6"}
    
    result = _pack_into_return_datatypes(
        all_completions, all_prompts, all_numbers,
        all_targets, all_tags, all_solutions
    )
    
    assert len(result) == 1
    assert len(result[0].branched_completions) == 3
    completions = [bc.branches[0].completion for bc in result[0].branched_completions]
    assert "First completion" in completions
    assert "Second completion" in completions
    assert "Third completion" in completions


def test_pack_into_return_datatypes_with_branches():
    """Test packing completions with branching structure."""
    all_completions = {
        "1#0_0_0_0_0": "Base completion",
        "1#0_0_1_0_0": "Branch 1",
        "1#0_0_2_0_0": "Branch 2",
        "1#0_0_1_1_0": "Sub-branch"
    }
    all_prompts = {"1#0": "Branching test"}
    all_numbers = {"1#0": [5, 5]}
    all_targets = {"1#0": 10}
    all_tags = {"1#0": "branch"}
    all_solutions = {"1#0": "10"}
    
    result = _pack_into_return_datatypes(
        all_completions, all_prompts, all_numbers,
        all_targets, all_tags, all_solutions
    )
    
    assert len(result) == 1
    # All branches belong to same base completion (1#0_0)
    assert len(result[0].branched_completions) == 1
    assert len(result[0].branched_completions[0].branches) == 4


def test_pack_into_return_datatypes_multiple_prompts():
    """Test packing multiple prompts with their completions."""
    all_completions = {
        "1#0_0_0_0_0": "Completion for prompt 1",
        "2#0_0_0_0_0": "Completion for prompt 2",
        "2#0_1_0_0_0": "Second completion for prompt 2"
    }
    all_prompts = {"1#0": "Prompt 1", "2#0": "Prompt 2"}
    all_numbers = {"1#0": [1], "2#0": [2]}
    all_targets = {"1#0": 1, "2#0": 2}
    all_tags = {"1#0": "tag1", "2#0": "tag2"}
    all_solutions = {"1#0": "sol1", "2#0": "sol2"}
    
    result = _pack_into_return_datatypes(
        all_completions, all_prompts, all_numbers,
        all_targets, all_tags, all_solutions
    )
    
    assert len(result) == 2
    # Check that we have the right prompts
    prompts = {pc.prompt for pc in result}
    assert prompts == {"Prompt 1", "Prompt 2"}
    
    # Find prompt 2 and verify it has 2 completions
    for pc in result:
        if pc.prompt == "Prompt 2":
            assert len(pc.branched_completions) == 2


def test_pack_into_return_datatypes_with_bare_prompts():
    """Test packing with optional bare prompts."""
    all_completions = {"1#0_0_0_0_0": "Completion"}
    all_prompts = {"1#0": "Full prompt with formatting"}
    all_numbers = {"1#0": [3, 3]}
    all_targets = {"1#0": 6}
    all_tags = {"1#0": "test"}
    all_solutions = {"1#0": "6"}
    all_bare_prompts = {"1#0": "Bare prompt"}
    
    result = _pack_into_return_datatypes(
        all_completions, all_prompts, all_numbers,
        all_targets, all_tags, all_solutions, all_bare_prompts
    )
    
    assert len(result) == 1
    assert result[0].bare_prompt == "Bare prompt"
    assert result[0].prompt == "Full prompt with formatting"


def test_pack_into_return_datatypes_branch_keys():
    """Test that branch keys are preserved correctly."""
    all_completions = {
        "123#5_0_1_2_3": "Completion with specific key"
    }
    all_prompts = {"123#5": "Test"}
    all_numbers = {"123#5": [1]}
    all_targets = {"123#5": 1}
    all_tags = {"123#5": "test"}
    all_solutions = {"123#5": "1"}
    
    result = _pack_into_return_datatypes(
        all_completions, all_prompts, all_numbers,
        all_targets, all_tags, all_solutions
    )
    
    assert len(result) == 1
    branch = result[0].branched_completions[0].branches[0]
    assert branch.key == "123#5_0_1_2_3"
    assert branch.completion == "Completion with specific key"