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

    def get_avg_error_func(self, transformations):
        """
            Returns function for applying the desired transformations to string and returning the average number of errors.
        """

        def avg_error_func(x):
            for transformation in transformations:
                x = transformation(x)
            return self.count_phrase_errors(x)

        if transformations is None:
            identity_func = lambda x: x
            identity_func.__name__ = 'identity'
            transformations = [identity_func]

        if type(transformations) != list:
            transformations = [transformations]

        transformtions_names = '-'.join([transformation.__name__ for transformation in transformations])
        avg_error_func.__name__ = 'avg_error_func' + '_transformations-' + transformtions_names
        return avg_error_func

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
