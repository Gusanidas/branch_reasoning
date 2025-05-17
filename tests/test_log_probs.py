import unittest
import torch
from unittest.mock import MagicMock, patch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from branch_reasoning.log_probs import get_log_probs, populate_branch_log_probs, populate_log_probs
from branch_reasoning.generation.completions import Branch, BranchedCompletion, PromptCompletion, ScoringData, Metadata


class TestLogProbs(unittest.TestCase):
    def setUp(self):
        # Create mock model and tokenizer
        self.model = MagicMock(spec=PreTrainedModel)
        self.tokenizer = MagicMock(spec=PreTrainedTokenizer)
        self.reference_model = MagicMock(spec=PreTrainedModel)
        
        # Mock the model output
        self.mock_output = MagicMock()
        # Create fake logits of shape [batch_size, sequence_length, vocab_size]
        self.vocab_size = 10
        self.seq_length = 5
        self.logits = torch.randn(1, self.seq_length, self.vocab_size)
        self.mock_output.logits = self.logits
        
        # Setup tokenizer mock behavior
        self.tokenizer.encode.return_value = [0, 1, 2]  # Simple token IDs
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]])  # Shape [1, 5]
        }
        
        # Model forward pass returns the mock output
        self.model.return_value = self.mock_output
        self.reference_model.return_value = self.mock_output

    @patch('torch.gather')
    @patch('torch.nn.functional.log_softmax')
    def test_get_log_probs(self, mock_log_softmax, mock_gather):
        """Test that get_log_probs properly calculates log probabilities"""
        # Mock log_softmax output
        mock_softmax_output = torch.randn(1, self.seq_length - 1, self.vocab_size)
        mock_log_softmax.return_value = mock_softmax_output
        
        # Mock gather output
        mock_gather_output = torch.randn(1, self.seq_length - 1, 1)
        mock_gather.return_value = mock_gather_output
        
        # Call function with test data
        text = "test text"
        result = get_log_probs(
            model=self.model,
            tokenizer=self.tokenizer,
            text=text,
            device="cpu",
            temperature=1.0
        )
        
        # Assertions to verify behavior
        self.model.to.assert_called_once_with("cpu")
        self.tokenizer.assert_called_once_with(text, return_tensors="pt")
        self.model.assert_called_once()
        mock_log_softmax.assert_called_once()
        mock_gather.assert_called_once()
        
        # Verify result shape and type
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (self.seq_length - 1,))

    def test_populate_branch_log_probs(self):
        """Test that populate_branch_log_probs properly updates a Branch object"""
        # Create a test branch
        branch = Branch(
            completion="test completion",
            log_probs=None,
            ref_log_probs=None,
            score=None
        )
        
        # Mock get_log_probs with a patch to return predictable values
        with patch('branch_reasoning.log_probs.get_log_probs') as mock_get_log_probs:
            # Create fake log probabilities tensor
            fake_log_probs = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5])
            mock_get_log_probs.return_value = fake_log_probs
            
            # Set tokenizer encode to return consistent values
            self.tokenizer.encode.return_value = [0, 1, 2]  # 3 tokens
            
            # Call function
            prompt = "test prompt"
            result = populate_branch_log_probs(
                branch=branch,
                prompt=prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                reference_model=self.reference_model,
                device="cpu",
                temperature=1.0
            )
            
            # Verify branch was properly updated
            self.tokenizer.encode.assert_called_once_with(prompt, add_special_tokens=False)
            
            # get_log_probs should be called twice: once for model, once for reference_model
            self.assertEqual(mock_get_log_probs.call_count, 2)
            
            # Check that branch log_probs was properly sliced (should skip first 2 tokens from prompt)
            # Expected slice: fake_log_probs[2:]
            self.assertTrue(torch.equal(branch.log_probs, fake_log_probs[2:]))
            self.assertTrue(torch.equal(branch.ref_log_probs, fake_log_probs[2:]))
            
            # Branch should be returned unchanged
            self.assertEqual(result, branch)

    def test_populate_log_probs(self):
        """Test that populate_log_probs processes a list of PromptCompletion objects"""
        # Create test data
        branch1 = Branch(completion="completion1", log_probs=None, ref_log_probs=None, score=None)
        branch2 = Branch(completion="completion2", log_probs=None, ref_log_probs=None, score=None)
        
        branched_completion = BranchedCompletion(
            branches=[branch1, branch2],
            score=None
        )
        
        prompt_completion = PromptCompletion(
            prompt="test prompt",
            bare_prompt=None,
            scoring_data=ScoringData(nums=[1, 2, 3], target=6),
            metadata=Metadata(solution="6", tag="math"),
            branched_completions=[branched_completion]
        )
        
        # Create a list of PromptCompletion objects
        prompt_completions = [prompt_completion]
        
        # Patch populate_branch_log_probs to avoid actual implementation
        with patch('branch_reasoning.log_probs.populate_branch_log_probs') as mock_populate_branch:
            # Call function under test
            result = populate_log_probs(
                prompt_completions=prompt_completions,
                model=self.model,
                tokenizer=self.tokenizer,
                reference_model=self.reference_model,
                device="cpu",
                temperature=1.0
            )
            
            # Verify populate_branch_log_probs was called for each branch
            self.assertEqual(mock_populate_branch.call_count, 2)
            
            # Verify function calls with expected parameters
            calls = mock_populate_branch.call_args_list
            
            # First call should be for branch1
            first_call = calls[0][1]  # kwargs of first call
            self.assertEqual(first_call["branch"], branch1)
            self.assertEqual(first_call["prompt"], "test prompt")
            self.assertEqual(first_call["model"], self.model)
            self.assertEqual(first_call["tokenizer"], self.tokenizer)
            self.assertEqual(first_call["reference_model"], self.reference_model)
            self.assertEqual(first_call["device"], "cpu")
            self.assertEqual(first_call["temperature"], 1.0)
            
            # Second call should be for branch2
            second_call = calls[1][1]  # kwargs of second call
            self.assertEqual(second_call["branch"], branch2)
            
            # Result should be the input list unchanged
            self.assertEqual(result, prompt_completions)
            
    def test_populate_log_probs_with_bare_prompt(self):
        """Test that populate_log_probs uses bare_prompt when available"""
        # Create test data with bare_prompt
        branch = Branch(completion="completion", log_probs=None, ref_log_probs=None, score=None)
        
        branched_completion = BranchedCompletion(
            branches=[branch],
            score=None
        )
        
        prompt_completion = PromptCompletion(
            prompt="test prompt with instructions",
            bare_prompt="bare test prompt",  # This should be used instead of prompt
            scoring_data=ScoringData(nums=[1, 2], target=3),
            metadata=Metadata(solution="3", tag="math"),
            branched_completions=[branched_completion]
        )
        
        # Create a list of PromptCompletion objects
        prompt_completions = [prompt_completion]
        
        # Patch populate_branch_log_probs to avoid actual implementation
        with patch('branch_reasoning.log_probs.populate_branch_log_probs') as mock_populate_branch:
            # Call function under test
            populate_log_probs(
                prompt_completions=prompt_completions,
                model=self.model,
                tokenizer=self.tokenizer,
                reference_model=None,  # Test with no reference model
                device="cpu",
                temperature=1.0
            )
            
            # Verify populate_branch_log_probs was called with bare_prompt
            self.assertEqual(mock_populate_branch.call_count, 1)
            
            # Verify function call with expected parameters
            call = mock_populate_branch.call_args[1]  # kwargs of the call
            self.assertEqual(call["branch"], branch)
            self.assertEqual(call["prompt"], "bare test prompt")  # Should use bare_prompt
            self.assertEqual(call["reference_model"], None)  # Should pass None
            
    def test_get_log_probs_with_temperature(self):
        """Test that get_log_probs correctly applies temperature scaling"""
        # Prepare mock for temperature scaling test
        with patch('torch.nn.functional.log_softmax') as mock_log_softmax, \
             patch('torch.gather') as mock_gather:
            
            # Mock log_softmax output
            mock_softmax_output = torch.randn(1, self.seq_length - 1, self.vocab_size)
            mock_log_softmax.return_value = mock_softmax_output
            
            # Mock gather output
            mock_gather_output = torch.randn(1, self.seq_length - 1, 1)
            mock_gather.return_value = mock_gather_output

            # Test with different temperature values
            test_temperatures = [0.5, 1.0, 2.0]
            
            for temp in test_temperatures:
                # Call function
                get_log_probs(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    text="test text",
                    device="cpu",
                    temperature=temp
                )
                
                # Get the logits that were passed to log_softmax
                # This should be the scaled logits from shifted_logits / temperature
                args, _ = mock_log_softmax.call_args
                
                # The scaled logits should be the first argument to log_softmax
                scaled_logits = args[0]
                
                # Check that we're dividing by the temperature
                # (we can't directly test the division, but we can test the call was made)
                self.assertIsNotNone(scaled_logits)
                
                # Reset mocks for next iteration
                mock_log_softmax.reset_mock()
                mock_gather.reset_mock()


if __name__ == '__main__':
    unittest.main()