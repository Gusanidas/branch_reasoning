import unittest
from unittest.mock import MagicMock, patch
from branch_reasoning.generation.text_generation import hf_generate_text

class TestTextGeneration(unittest.TestCase):
    @patch('branch_reasoning.generation.text_generation.pipeline')
    def test_hf_generate_text_basic(self, mock_pipeline):
        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_generator = MagicMock()
        mock_pipeline.return_value = mock_generator
        
        # Set up mock pipeline output
        mock_generator.return_value = [{'generated_text': 'Test output 1'}, {'generated_text': 'Test output 2'}]
        
        # Test function
        results = hf_generate_text(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompts=["Test prompt"],
            num_completions=2,
            max_seq_len=100,
            device="cpu"
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "Test output 1")
        self.assertEqual(results[1], "Test output 2")
        
        # Verify pipeline was called correctly
        mock_pipeline.assert_called_once()
        mock_generator.assert_called_once()

if __name__ == "__main__":
    unittest.main()
