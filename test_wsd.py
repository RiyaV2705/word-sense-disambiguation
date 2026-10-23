import unittest
from wsd import tokenize

class TestHelpers(unittest.TestCase):
    def test_tokenize_lowercases(self):
        self.assertIn("money", tokenize("Money is in the BANK."))

    def test_tokenize_removes_common_stopwords(self):
        tokens = tokenize("The cat is on the mat")
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)

    def test_tokenize_handles_punctuation(self):
        self.assertEqual(tokenize("river-bank, trees!"), ["river", "bank", "trees"])

if __name__ == "__main__":
    unittest.main()
