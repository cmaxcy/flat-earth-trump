import json
import language_check
from parse_tools import *
import itertools

class Grammar:

    def __init__(self, phrase_cache_path=None):

        if phrase_cache_path is None:
            self.phrase_cache = dict()
        else:
            self.phrase_cache = self.load_phrase_cache(phrase_cache_path)

        self.tool = language_check.LanguageTool('en-US')

    def write_phrase_cache(self, phrase_cache_path):
        """
            Wite phrase cache to json file.
        """

        # Dump dictionary to json
        dumped = json.dumps(self.phrase_cache)

        # Write json to file
        with open(phrase_cache_path, "w") as text_file:
            text_file.write(dumped)

    @staticmethod
    def cartesian_product(*args):
        return list(itertools.product(*args))

    @staticmethod
    def sequify_string(string, n, only_words=True, transformations=None):
        """
            Returns all word sequences in string of length n in the order in which
            they occur. Word sequences are returned as full strings. If only words is specified, punctuation and other
            symbols will be ignored. Transformations will be applied to string in the order in which they appear
            before process is started (identity transformation applied by default).
        """

        if transformations is None:
            transformations = [lambda x: x]

        if type(transformations) != list:
            transformations = [transformations]

        for transformation in transformations:
            string = transformation(string)

        if only_words:
            string_pieces = extract_words(string)
        else:
            string_pieces = string.split()

        sequences = Grammar.n_length_sequences(string_pieces, n)
        return list(map(' '.join, sequences))

    @staticmethod
    def label_apply(data, functions):
        """
            Maps functions to data and stores in dataframe with column names
            representing name of applied functions.
        """

        df = pd.DataFrame()

        df['data'] = data

        for function in functions:
            applied = list(map(function, data))
            df[function.__name__] = applied

        return df

    def get_avg_error_func(self, n, only_words, transformations):
        """
            Returns function for applying the desired transformations to string and returning the average number of errors.
        """

        def avg_error_func(x):
            if len(self.sequify_string(x, n, only_words, transformations)) == 0:
                return 0

            return sum(map(self.count_phrase_errors, self.sequify_string(x, n, only_words, transformations))) / len(self.sequify_string(x, n, only_words, transformations))

        if transformations is None:
            identity_func = lambda x: x
            identity_func.__name__ = 'identity'
            transformations = [identity_func]

        if type(transformations) != list:
            transformations = [transformations]

        transformtions_names = '-'.join([transformation.__name__ for transformation in transformations])

        avg_error_func.__name__ = 'avg_error_func_n-' + str(n) + '_only_words-' + str(only_words) + '_transformations-' + transformtions_names
        return avg_error_func

    @staticmethod
    def n_length_sequences(base_list, n):
        """
            Returns all contiguous sequences of length n in list.

            Credit: https://stackoverflow.com/questions/6670828/find-all-consecutive-sub-sequences-of-length-n-in-a-sequence
        """
        if n < 1 or n > len(base_list):
            return []

        return list(map(list, zip(*(base_list[i:] for i in range(n)))))

    def count_phrase_errors(self, string):
        """
            Returns the number of errors found when a grammar check is performed on the string.

            Results of check are cached, and will be attempted to be read before check is made.
        """
        if string in self.phrase_cache:
            return self.phrase_cache[string]

        grammar_tool_result = len(self.tool.check(string))
        self.phrase_cache[string] = grammar_tool_result

        return grammar_tool_result

    # TODO:
    # - Consider verifying that path passed is phrase cache (keys are strings, values are integers, etc.)
    @staticmethod
    def load_phrase_cache(json_path):
        """
            Read phrase cache from json file.
        """
        with open(json_path, "r") as text_file:
            return json.loads(text_file.read())
