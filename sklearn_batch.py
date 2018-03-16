import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import pickle
import shutil

class SklearnBatch:

    # TODO:
    # - Consider what to do when name passed already has folder (prevent?)
    def __init__(self, name, feature_column_names, label_column_name, model_dict=None):
        self.name = name
        self.feature_column_names = feature_column_names
        self.label_column_name = label_column_name
        if model_dict is None:
            self.model_dict = dict()
        else:
            self.model_dict = model_dict

    def predict(self, examples, data_column_name):

        # Create custom copy of argument
        examples = pd.DataFrame(examples, copy=True)

        output = pd.DataFrame()

        output[data_column_name] = examples[data_column_name]
        del examples[data_column_name]

        X = examples

        for model_name in sorted(self.model_dict.keys()):
            model = self.model_dict[model_name]
            y = model.decision_function(X)
            output[model_name] = y

        return output

    def train_new(self, model_name, examples, text_column_name, model=None):

        if model_name in self.model_dict:
            raise ValueError('Model name already in use in dictionary')

        if model is None:
            model = GradientBoostingClassifier()

        # Create custom copy of argument
        examples = pd.DataFrame(examples, copy=True)

        del examples[text_column_name]

        y = examples[self.label_column_name]
        del examples[self.label_column_name]
        X = examples

        model.fit(X, y)

        self.model_dict[model_name] = model

    def write_to_folder(self, clobber=True):

        # If path already exists, clobber or break
        if os.path.exists(self.name):
            if clobber:
                shutil.rmtree(self.name)
            else:
                raise ValueError('Folder with name ' + self.name + ' alrady exists')

        # Create folder with instance name
        os.makedirs(self.name)

        # Store instance info in dictionary
        info_dict = dict()
        info_dict['label'] = self.label_column_name
        info_dict['features'] = self.feature_column_names

        # Write dictionary to file
        write_path = os.path.join(self.name, 'column_names.pkl')
        with open(write_path, 'wb') as fp:
            pickle.dump(info_dict, fp)

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

        # Ensure directory contains file with column names
        column_names_file_path = os.path.join(name, 'column_names.pkl')
        if not os.path.isfile(column_names_file_path):
            raise ValueError('No column_names.pkl in direcory')

        # Retrieve column data
        with open(column_names_file_path, 'rb') as fp:
            info_dict = pickle.load(fp)
        label_column_name = info_dict['label']
        feature_column_names = info_dict['features']

        # Collect stored models into dictionary
        model_dict = dict()
        model_paths = os.listdir(name)
        model_paths.remove('column_names.pkl')
        for model_path in map(lambda x: os.path.join(name, x), model_paths):
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            model_dict[model_name] = joblib.load(model_path)

        return SklearnBatch(name=name, feature_column_names=feature_column_names, label_column_name=label_column_name, model_dict=model_dict)
