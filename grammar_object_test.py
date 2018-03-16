from grammar_object import Grammar
import json
import unittest
import os
import time
from parse_tools import *

class TestGrammar(unittest.TestCase):

    def test_label_apply(self):

        def test_func_1(arg):
            return 1 * arg

        def test_func_2(arg):
            return 2 * arg

        def test_func_3(arg):
            return 3 * arg

        test_functions = [test_func_1, test_func_2, test_func_3]
        test_data = [1, 2, 3, 4]

        expected_output = pd.DataFrame({
            'data': test_data,
            'test_func_1': [1, 2, 3, 4],
            'test_func_2': [2, 4, 6, 8],
            'test_func_3': [3, 6, 9, 12],
        })

        actual_output = Grammar.label_apply(test_data, test_functions)

        pd.testing.assert_frame_equal(expected_output, actual_output)

    def test_cartesian_product_lists(self):
        """
            Verify that cartesian product can be calculated with lists of items.
        """
        self.assertEqual(Grammar.cartesian_product(['a', 'b'], [1, 2]), [('a', 1), ('a', 2), ('b', 1), ('b', 2)])

    def test_cartesian_product_single_items(self):
        """
            Verify that cartesian product can be calculated with single items.
        """

        # Arguments must be iterables
        self.assertRaises(TypeError, Grammar.cartesian_product, 'a', 1)

        self.assertEqual(Grammar.cartesian_product(['a'], [1]), [('a', 1)])

        # Strings are iterables
        self.assertEqual(Grammar.cartesian_product('a', 'b'), [('a', 'b')])

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

    def test_get_avg_error_function_n(self):
        """
            Verify that get_avg_error_func can correctly produce a function
            that obtains the average number of grammar errors in the sequences
            of size n in the string.

            Verify behavior with multiple n arguments.
        """

        test_string = "This is nothing but a test string"

        # Obtain testing average error function
        test_grammar = Grammar()
        test_n_arg = 2
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        # Manually re-create function
        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        # Verify error on each sequence
        self.assertEqual(sequified_string[0], "This is")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)

        self.assertEqual(sequified_string[1], "is nothing")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[1]), 1)

        self.assertEqual(sequified_string[2], "nothing but")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[2]), 1)

        self.assertEqual(sequified_string[3], "but a")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[3]), 1)

        self.assertEqual(sequified_string[4], "a test")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[4]), 1)

        self.assertEqual(sequified_string[5], "test string")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[5]), 1)

        # 5 total errors observed, so average error is 5/6
        self.assertEqual(test_func(test_string), 5/6)

        # Repeat process with n = 4
        test_string = "This is nothing but a test string"
        test_grammar = Grammar()
        test_n_arg = 4
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        self.assertEqual(sequified_string[0], "This is nothing but")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)
        self.assertEqual(sequified_string[1], "is nothing but a")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[1]), 1)
        self.assertEqual(sequified_string[2], "nothing but a test")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[2]), 1)
        self.assertEqual(sequified_string[3], "but a test string")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[3]), 1)

        self.assertEqual(test_func(test_string), 3/4)

        # Repeat process with n = 7
        test_string = "This is nothing but a test string"
        test_grammar = Grammar()
        test_n_arg = 7
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        self.assertEqual(sequified_string[0], "This is nothing but a test string")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)

        self.assertEqual(test_func(test_string), 0)


    def test_get_avg_error_function_only_words(self):
        """
            Verify that get_avg_error_func can correctly produce a function
            that obtains the average number of grammar errors in the sequences
            of size n in the string.

            Verify behavior with multiple only_words arguments.
        """

        test_string = "Here is a string. Two sentences are present."

        # Obtain testing average error function
        test_grammar = Grammar()
        test_n_arg = 5
        test_only_words_arg = False
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        # Manually re-create function
        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        # Verify error on each sequence
        self.assertEqual(sequified_string[0], "Here is a string. Two")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)

        self.assertEqual(sequified_string[1], "is a string. Two sentences")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[1]), 1)

        self.assertEqual(sequified_string[2], "a string. Two sentences are")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[2]), 1)

        self.assertEqual(sequified_string[3], "string. Two sentences are present.")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[3]), 1)

        # 3 total errors observed, so average error is 3/4
        self.assertEqual(test_func(test_string), 3/4)

        # Repeat process with only_words = True
        test_string = "Here is a string. Two sentences are present."
        test_grammar = Grammar()
        test_n_arg = 5
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        self.assertEqual(sequified_string[0], "Here is a string Two")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)
        self.assertEqual(sequified_string[1], "is a string Two sentences")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[1]), 1)
        self.assertEqual(sequified_string[2], "a string Two sentences are")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[2]), 1)
        self.assertEqual(sequified_string[3], "string Two sentences are present")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[3]), 1)

        self.assertEqual(test_func(test_string), 3/4)

    def test_get_avg_error_function_transformations(self):
        """
            Verify that get_avg_error_func can correctly produce a function
            that obtains the average number of grammar errors in the sequences
            of size n in the string.

            Verify behavior with different transformations on string.
        """

        test_string = "Here is a string with an at: @realDonaldTrump"

        # Obtain testing average error function
        test_grammar = Grammar()
        test_n_arg = 3
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        # Manually re-create function
        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        # Verify error on each sequence
        self.assertEqual(sequified_string[0], "Here is a")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)

        self.assertEqual(sequified_string[1], "is a string")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[1]), 1)

        self.assertEqual(sequified_string[2], "a string with")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[2]), 1)

        self.assertEqual(sequified_string[3], "string with an")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[3]), 1)

        self.assertEqual(sequified_string[4], "with an at")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[4]), 1)

        self.assertEqual(sequified_string[5], "an at @realDonaldTrump")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[5]), 2)

        # 6 total errors observed, so average error is 1
        self.assertEqual(test_func(test_string), 1)

        # Repeat process with @ replacement transformation
        test_string = "Here is a string with an at: @realDonaldTrump"
        test_grammar = Grammar()
        test_n_arg = 3
        test_only_words_arg = True
        test_transformations_arg = lambda x: replace_ats(x, "John")
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        self.assertEqual(sequified_string[0], "Here is a")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)
        self.assertEqual(sequified_string[1], "is a string")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[1]), 1)
        self.assertEqual(sequified_string[2], "a string with")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[2]), 1)
        self.assertEqual(sequified_string[3], "string with an")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[3]), 1)
        self.assertEqual(sequified_string[4], "with an at")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[4]), 1)
        self.assertEqual(sequified_string[5], "an at John")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[5]), 1)

        self.assertEqual(test_func(test_string), 5/6)

        # Repeat process with an additional hastag replacement transformation
        test_string = "Here is a string with an at: @realDonaldTrump and a hashtag: #MakeAmericaGreatAgain"
        test_grammar = Grammar()
        test_n_arg = 3
        test_only_words_arg = True
        test_transformations_arg = [lambda x: replace_ats(x, "John"), lambda x: remove_hts(x)]
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        self.assertEqual(sequified_string[0], "Here is a")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[0]), 0)
        self.assertEqual(sequified_string[1], "is a string")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[1]), 1)
        self.assertEqual(sequified_string[2], "a string with")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[2]), 1)
        self.assertEqual(sequified_string[3], "string with an")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[3]), 1)
        self.assertEqual(sequified_string[4], "with an at")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[4]), 1)
        self.assertEqual(sequified_string[5], "an at John")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[5]), 1)
        self.assertEqual(sequified_string[6], "at John and")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[6]), 1)
        self.assertEqual(sequified_string[7], "John and a")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[7]), 0)
        self.assertEqual(sequified_string[8], "and a hashtag")
        self.assertEqual(test_grammar.count_phrase_errors(sequified_string[8]), 2)

        self.assertEqual(test_func(test_string), 8/9)

    def test_get_avg_error_function_naming(self):
        """
            Verify that get_avg_error_func can correctly produce a function
            that obtains the average number of grammar errors in the sequences
            of size n in the string.

            Verify that created function is named appropriatly.
        """
        test_grammar = Grammar()

        test_n_arg = 3
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)
        self.assertEqual(test_func.__name__, "avg_error_func_n-3_only_words-True_transformations-identity")

        test_n_arg = 32
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)
        self.assertEqual(test_func.__name__, "avg_error_func_n-32_only_words-True_transformations-identity")

        test_n_arg = 3
        test_only_words_arg = False
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)
        self.assertEqual(test_func.__name__, "avg_error_func_n-3_only_words-False_transformations-identity")

        test_n_arg = 3
        test_only_words_arg = True
        test_transformations_arg = [remove_hts]
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)
        self.assertEqual(test_func.__name__, "avg_error_func_n-3_only_words-True_transformations-remove_hts")

        test_n_arg = 3
        test_only_words_arg = True
        test_transformations_arg = remove_hts
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)
        self.assertEqual(test_func.__name__, "avg_error_func_n-3_only_words-True_transformations-remove_hts")

        test_n_arg = 3
        test_only_words_arg = True
        test_transformations_arg = [remove_hts, remove_ats]
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)
        self.assertEqual(test_func.__name__, "avg_error_func_n-3_only_words-True_transformations-remove_hts-remove_ats")

    # TODO:
    # - Consider case with null string
    def test_get_avg_error_function_alt_arguments(self):
        """
            Verify that get_avg_error_func can correctly produce a function
            that obtains the average number of grammar errors in the sequences
            of size n in the string.

            Verify that arguments that do not go together (n longer than the string passed, etc.)
            are handled correctly.
        """

        # Test with n larger than number of words in string
        test_string = "This string has five words"
        test_grammar = Grammar()
        test_n_arg = 6
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        # Because no string sequences should have been generated, no errors will be present
        self.assertEqual(test_func(test_string), 0)

        # Test with n = 0
        test_string = "This string has five words"
        test_grammar = Grammar()
        test_n_arg = 0
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        # Because no string sequences should have been generated, no errors will be present
        self.assertEqual(test_func(test_string), 0)

        # Test with n = -1
        test_string = "This string has five words"
        test_grammar = Grammar()
        test_n_arg = -1
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        # Because no string sequences should have been generated, no errors will be present
        self.assertEqual(test_func(test_string), 0)

        # Test with null string
        test_string = ""
        test_grammar = Grammar()
        test_n_arg = -1
        test_only_words_arg = True
        test_transformations_arg = None
        test_func = test_grammar.get_avg_error_func(test_n_arg, test_only_words_arg, test_transformations_arg)

        sequified_string = test_grammar.sequify_string(string=test_string, n=test_n_arg, only_words=test_only_words_arg, transformations=test_transformations_arg)

        # Because no string sequences should have been generated, no errors will be present
        self.assertEqual(test_func(test_string), 0)

    def test_sequify_string(self):
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_string = "Another sample string"
        sequence_lengths = 3
        expected_sequified_string = ["Another sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_string = "One more"
        sequence_lengths = 1
        expected_sequified_string = ["One", "more"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

    def test_sequify_string_word_options(self):
        """
            Verify behavior of sequify_string with word keeping options.
        """
        test_string = "Here is a sample sentence."
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample sentence"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, only_words=True)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_string = "Here is a sample sentence."
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample sentence."]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, only_words=False)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_string = "Here is a sample sentence. Here is another"
        sequence_lengths = 4
        expected_sequified_string = ["Here is a sample", "is a sample sentence", "a sample sentence Here", "sample sentence Here is", "sentence Here is another"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, only_words=True)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_string = "Here is a sample sentence. Here is another"
        sequence_lengths = 4
        expected_sequified_string = ["Here is a sample", "is a sample sentence.", "a sample sentence. Here", "sample sentence. Here is", "sentence. Here is another"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, only_words=False)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

    def test_sequify_string_empty_sequence_outcomes(self):
        """
            Verify that empty sequences are produced on arguents where the sequence size is less than 1, or larger exceeds that which is possible with the string provided.
        """
        test_string = "Here is a sample string"
        sequence_lengths = 0
        expected_sequified_string = []
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_string = "Here is a sample string"
        sequence_lengths = 6
        expected_sequified_string = []
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

    def test_sequify_string_transformations(self):
        """
            Verify that transformations can be applied to the string and that sequences reflect this transformation.
        """
        test_transformations = lambda x: ""

        # transformation NOT applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=None)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        # transformation applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = []
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=test_transformations)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_transformations = [lambda x: ""]

        # transformation NOT applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=None)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        # transformation applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = []
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=test_transformations)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_transformations = [lambda x: x + " phrase addition"]

        # transformation NOT applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=None)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        # transformation applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string", "string phrase", "phrase addition"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=test_transformations)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        test_transformations = [lambda x: x + " phrase", lambda x: x + " addition"]

        # transformation NOT applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=None)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        # transformation applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string", "string phrase", "phrase addition"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=test_transformations)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

    def test_sequify_string_invalid_transformations(self):
        """
            Verify that invalid transoformations are handled appropriatley.
        """
        test_transformations = [lambda x: 1]

        # transformation NOT applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=None)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        # transformation applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        self.assertRaises(TypeError, Grammar.sequify_string, test_string, sequence_lengths, transformations=test_transformations)

        test_transformations = [lambda x: 1, lambda x: x + ""]

        # transformation NOT applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=None)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        # transformation applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        self.assertRaises(TypeError, Grammar.sequify_string, test_string, sequence_lengths, transformations=test_transformations)

        test_transformations = [lambda x: x + "", lambda x: 1]

        # transformation NOT applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        expected_sequified_string = ["Here is", "is a", "a sample", "sample string"]
        actual_sequified_string = Grammar.sequify_string(test_string, sequence_lengths, transformations=None)
        self.assertEqual(expected_sequified_string, actual_sequified_string)

        # transformation applied
        test_string = "Here is a sample string"
        sequence_lengths = 2
        self.assertRaises(TypeError, Grammar.sequify_string, test_string, sequence_lengths, transformations=test_transformations)

    def test_n_length_sequences_invalid(self):
        """
            Verify behavior of n_length_sequences on invalid arguments.
        """
        self.assertEqual(Grammar.n_length_sequences([1, 2, 3], 4), [])
        self.assertEqual(Grammar.n_length_sequences([], 0), [])
        self.assertEqual(Grammar.n_length_sequences([], -1), [])

    def test_n_length_sequences_valid(self):
        """
            Verify behavior of n_length_sequences on valid lists and sequence sizes.
        """
        self.assertEqual(Grammar.n_length_sequences([1, 2, 3], 3), [[1, 2, 3]])
        self.assertEqual(Grammar.n_length_sequences([1, 2, 3], 2), [[1, 2], [2, 3]])
        self.assertEqual(Grammar.n_length_sequences([1, 2, 3], 1), [[1], [2], [3]])

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
