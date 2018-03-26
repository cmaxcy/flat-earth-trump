from grammar_object import Grammar
import json
import unittest
import os
import time
from parse_tools import ParseTools

# TODO:
# - Test changes to get_avg_error_func

class TestGrammar(unittest.TestCase):

    def setUp(self):
        """
            Create a sample phrase cache json file.
        """
        sample_phrase_cache = {'Sample phrase': 0, 'Another sample phrase': 2}
        with open('sample_phrase_cache.json', "w") as text_file:
            text_file.write(json.dumps(sample_phrase_cache))

    def tearDown(self):
        """
            Remove sample phrase cache json file.
        """
        os.remove('sample_phrase_cache.json')

    def test_count_phrase_errors(self):
        """
            Verify that phrases can have their correct number of errors calculated.
        """
        test_grammar = Grammar()
        self.assertEqual(test_grammar.count_phrase_errors('A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'), 2)
        self.assertEqual(test_grammar.count_phrase_errors('A sentence with an error in the Hitchhiker’s Guide to the Galaxy'), 0)

    def test_phrase_cache_updated_after_count_phrase_errors(self):
        """
            Verify that after the phrase cache is updated after seeing a new phrase.
        """
        test_grammar = Grammar()
        self.assertNotIn('New phrase', test_grammar.phrase_cache)
        test_grammar.count_phrase_errors('New phrase')
        self.assertIn('New phrase', test_grammar.phrase_cache)

    def test_constructor_no_passed_cache(self):
        """
            Verify that phrase cache is set to an empty dictionary when no path to an existing phrase cache is passed.
        """
        test_grammar = Grammar()
        self.assertEqual(test_grammar.phrase_cache, dict())

    def test_constructor_passed_cach(self):
        """
            Verify that phrase chache is read from from file when passed.
        """
        test_grammar = Grammar('sample_phrase_cache.json')
        expected_read_cache = {'Sample phrase': 0, 'Another sample phrase': 2}
        actual_read_cache = test_grammar.phrase_cache
        self.assertEqual(actual_read_cache, expected_read_cache)

    def test_load_phrase_cache_valid_path(self):
        """
            Verify that phrase cache can be loaded from json file.
        """
        expected_loaded_phrase_cache = {'Sample phrase': 0, 'Another sample phrase': 2}
        actual_loaded_phrase_cache = Grammar.load_phrase_cache('sample_phrase_cache.json')
        self.assertEqual(expected_loaded_phrase_cache, actual_loaded_phrase_cache)

if __name__ == "__main__":
    unittest.main()
