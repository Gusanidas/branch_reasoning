import unittest
from branch_reasoning.generation.branching import create_keyword_regex, find_subarray_nth_occurrence

class TestBranching(unittest.TestCase):
    def test_create_keyword_regex(self):
        pattern = create_keyword_regex(2)
        self.assertIsNotNone(pattern)
        self.assertTrue(pattern.search("#a#"))
        self.assertTrue(pattern.search("#b#"))
        self.assertFalse(pattern.search("#c#"))
        
        pattern = create_keyword_regex(3)
        self.assertIsNotNone(pattern)
        self.assertTrue(pattern.search("#a#"))
        self.assertTrue(pattern.search("#b#"))
        self.assertTrue(pattern.search("#c#"))
        self.assertFalse(pattern.search("#d#"))
        
    def test_find_subarray_nth_occurrence(self):
        text = "apple orange apple banana apple grape"
        self.assertEqual(find_subarray_nth_occurrence(text, "apple", 1), 0)
        self.assertEqual(find_subarray_nth_occurrence(text, "apple", 2), 13)
        self.assertEqual(find_subarray_nth_occurrence(text, "apple", 3), 26)
        self.assertEqual(find_subarray_nth_occurrence(text, "apple", 4), -1)
        self.assertEqual(find_subarray_nth_occurrence(text, "pear", 1), -1)

if __name__ == "__main__":
    unittest.main()
