import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle
import shutil
from grammar_object import Grammar
from parse_tools import ParseTools
from stat_tools import *
import itertools

class BatchGrammarClassifier:

    def __init__(self, name, model_dict=None, grammar=None):

        if grammar is None:
            grammar = Grammar()

        self.name = name

        transformation_functions = [ParseTools.remove_ats, ParseTools.remove_hts, ParseTools.replace_ats_with("John"), ParseTools.replace_ats_with("go")]
        trans_func_powerset = self.power_set(transformation_functions)
        self.grammar_functions = [grammar.get_avg_error_func(subset) for subset in trans_func_powerset]

        if model_dict is None:
            self.model_dict = dict()
        else:
            self.model_dict = model_dict

    @staticmethod
    def power_set(iterable):
        s = list(iterable)
        tuple_power_set = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
        listified = list(map(list, tuple_power_set))
        listified[0] = None
        return listified

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

    def predict(self, examples):

        grammared = self.label_apply(examples, self.grammar_functions)
        del grammared['data']

        output = pd.DataFrame()
        output['example'] = examples

        X = grammared

        for model_name in sorted(self.model_dict.keys()):
            model = self.model_dict[model_name]
            y = model.decision_function(X)
            output[model_name] = y

        return output

    @staticmethod
    def format_examples(negatives, positives):

        negatives['label'] = np.full(negatives.shape[0], 0)
        positives['label'] = np.full(positives.shape[0], 1)

        return pd.concat([negatives, positives], ignore_index=True)

    def train_new(self, model_name, negative_examples, positive_examples, grid_search=False):

        if model_name in self.model_dict:
            raise ValueError('Model name already in use in dictionary')

        # Grid search can be used to find optimum hyperparameters for gradient boosting
        if grid_search:
            parameters = {'learning_rate': [.01, .03, .1, .3, 1],
                          'n_estimators':[100, 300, 500, 1000],
                          'max_depth': [3, 4, 5, 6]}
            classifier = GradientBoostingClassifier()
            model = GridSearchCV(classifier, parameters)
        else:
            model = GradientBoostingClassifier()

        negatives = self.label_apply(negative_examples, self.grammar_functions)
        positives = self.label_apply(positive_examples, self.grammar_functions)
        examples = self.format_examples(negatives, positives)

        del examples['data']
        y = examples['label']
        del examples['label']
        X = examples

        model.fit(X, y)

        self.model_dict[model_name] = model

    # TODO: Document, Test
    def train_new_mimiced(self, model_name, examples, grid_search=False):
        mimiced_examples = self.mimic(examples)
        self.train_new(model_name, mimiced_examples, examples, grid_search)

    # TODO: Document
    @staticmethod
    def mimic(examples, punctuation_odds=.05):

        all_words = []
        example_word_counts = []

        for example in examples:
            example_words = ParseTools.extract_words(example)
            all_words += example_words
            example_word_counts.append(len(example_words))

        example_word_count_mean, example_word_count_std = np.mean(example_word_counts), np.std(example_word_counts)

        word_distribution = element_distribution(all_words)

        mimiced = []
        for _ in examples:
            sampled_length = np.clip(int(np.random.normal(example_word_count_mean, example_word_count_std)), a_min=1, a_max=None)

            mimiced_example = ""
            for word in sample(word_distribution, element_column_name='elements', probability_column_name='probabilities', n=sampled_length):
                for punctuation in {'.', '?', '!'}:
                    if np.random.uniform() <= punctuation_odds:
                        word += punctuation
                mimiced_example += word + " "
            mimiced_example = mimiced_example[:-1]

            mimiced.append(mimiced_example)

        return mimiced

    def write_to_folder(self, clobber=True):

        # If path already exists, clobber or break
        if os.path.exists(self.name):
            if clobber:
                shutil.rmtree(self.name)
            else:
                raise ValueError('Folder with name ' + self.name + ' alrady exists')

        # Create folder with instance name
        os.makedirs(self.name)

        # Write models to file
        for model_name in sorted(self.model_dict.keys()):
            model = self.model_dict[model_name]
            model_path = os.path.join(self.name, model_name + '.pkl')
            joblib.dump(model, model_path)

    @staticmethod
    def load_from_folder(name):

        # Ensure folder name is a directory
        if not os.path.isdir(name):
            raise ValueError('Non folder name passed')

        # Collect stored models into dictionary
        model_dict = dict()
        model_paths = os.listdir(name)
        for model_path in map(lambda x: os.path.join(name, x), model_paths):
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            model_dict[model_name] = joblib.load(model_path)

        return BatchGrammarClassifier(name=name, model_dict=model_dict)

    # TODO: Document
    def rank(self, examples):

        scores = self.predict(examples)
        del scores['example']

        summed = scores.sum(axis=1)
        totals = pd.DataFrame({
            'tweet': examples,
            'score': summed
        })

        return list(totals.sort_values('score', ascending=False)['tweet'])
