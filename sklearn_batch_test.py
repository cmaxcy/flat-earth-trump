import unittest
import pandas as pd
from sklearn_batch import *
import shutil
import os

# TODO:
# - Test clobber functonality of folder writing
class TestSklearnBatch(unittest.TestCase):

    def test_train_new_models_independent(self):
        '''
            Verify that when training a new model, the model is different than
            any existing ones.
        '''
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')

        test_new_model_name = 'model1'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_new_model_name = 'model2'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [1, 0, 1],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')

        model1 = test_sklearn_batch.model_dict['model1']
        model2 = test_sklearn_batch.model_dict['model2']

        self.assertNotEqual(model1, model2)


    def test_arguments_unchanged(self):
        '''
            Verify that methods do not alter their arguments.
        '''
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')

        test_new_model_name = 'model1'
        test_data_pre_pass = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data_pre_pass, 'text_column')
        expected_test_data_post_pass = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })

        # Verify argument is left unchanged
        pd.testing.assert_frame_equal(test_data_pre_pass, expected_test_data_post_pass)

        test_examples_pre_pass = pd.DataFrame({
            'feature1': [5, 6, 7],
            'feature2': [6, 12, 18],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_examples_post_pass = pd.DataFrame({
            'feature1': [5, 6, 7],
            'feature2': [6, 12, 18],
            'text_column': ['data1', 'data2', 'data3']
        })

        test_sklearn_batch.predict(test_examples_pre_pass, 'text_column')

        # Verify argument is left unchanged
        pd.testing.assert_frame_equal(test_examples_pre_pass, test_examples_post_pass)

    def test_predictions_consistent_after_load(self):
        '''
            Verify that model output is unchanged after instance is read and
            written to file.
        '''
        test_feature_column_names = ['feature1', 'feature2']
        test_label_column_name = 'label'
        test_name = 'test'
        test_sklearn_batch = SklearnBatch(name=test_name, feature_column_names=test_feature_column_names, label_column_name=test_label_column_name)

        test_new_model_name = 'model1'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_new_model_name = 'model2'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [1, 0, 1],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')

        test_examples = pd.DataFrame({
            'feature1': [5, 6, 7],
            'feature2': [6, 12, 18],
            'text_column': ['data1', 'data2', 'data3']
        })

        # Predictions before folder write
        test_pre_write_predictions = test_sklearn_batch.predict(test_examples, 'text_column')

        # Write then retrive instance
        test_sklearn_batch.write_to_folder()
        test_read_instance = SklearnBatch.load_from_folder(test_name)
        self.assertEqual(test_read_instance.name, test_name)
        self.assertEqual(test_read_instance.feature_column_names, test_feature_column_names)
        self.assertEqual(test_read_instance.label_column_name, test_label_column_name)

        # Predictions after folder write
        test_post_write_predictions = test_read_instance.predict(test_examples, 'text_column')
        pd.testing.assert_frame_equal(test_pre_write_predictions, test_post_write_predictions)

        # Delete created folder
        shutil.rmtree(test_name)

    def test_write_then_load(self):
        '''
            Verify that contents of SklearnBatch can be written and then read
            back from a folder.
        '''
        test_feature_column_names = ['feature1', 'feature2']
        test_label_column_name = 'label'
        test_name = 'test'
        test_sklearn_batch = SklearnBatch(name=test_name, feature_column_names=test_feature_column_names, label_column_name=test_label_column_name)
        test_sklearn_batch.write_to_folder()

        test_read_instance = SklearnBatch.load_from_folder(test_name)
        self.assertEqual(test_read_instance.name, test_name)
        self.assertEqual(test_read_instance.feature_column_names, test_feature_column_names)
        self.assertEqual(test_read_instance.label_column_name, test_label_column_name)

        # Delete created folder
        shutil.rmtree(test_name)

    def test_load_from_folder_non_existant_folder(self):
        '''
            Verify that a non-existant folder can be caught.
        '''
        with self.assertRaises(ValueError):
            SklearnBatch.load_from_folder('not_a_folder')

    def test_load_from_folder_no_column_name_file(self):
        '''
            Verify that a folder without a column name file can be caught.
        '''
        test_folder = 'tempfolder'
        os.makedirs(test_folder)
        with self.assertRaises(ValueError):
            SklearnBatch.load_from_folder(test_folder)
        # Delete created folder
        shutil.rmtree(test_folder)

    def test_write_no_clobber(self):

        # Build and write instance to folder
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')
        test_new_model_name = 'model1'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_new_model_name = 'model2'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [1, 0, 1],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_sklearn_batch.write_to_folder()
        self.assertTrue(os.path.isdir('test'))
        self.assertTrue(os.path.isfile('test/column_names.pkl'))
        self.assertTrue(os.path.isfile('test/model1.pkl'))
        self.assertTrue(os.path.isfile('test/model2.pkl'))

        with self.assertRaises(ValueError):
            test_sklearn_batch.write_to_folder(clobber=False)

        # Delete created folder
        shutil.rmtree('test')

    def test_write_clobber(self):

        # Build and write instance to folder
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')
        test_new_model_name = 'model1'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_new_model_name = 'model2'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [1, 0, 1],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_sklearn_batch.write_to_folder()
        self.assertTrue(os.path.isdir('test'))
        self.assertTrue(os.path.isfile('test/column_names.pkl'))
        self.assertTrue(os.path.isfile('test/model1.pkl'))
        self.assertTrue(os.path.isfile('test/model2.pkl'))

        test_new_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')
        test_new_batch.write_to_folder()
        self.assertTrue(os.path.isdir('test'))
        self.assertTrue(os.path.isfile('test/column_names.pkl'))
        self.assertFalse(os.path.isfile('test/model1.pkl'))
        self.assertFalse(os.path.isfile('test/model2.pkl'))

        # Delete created folder
        shutil.rmtree('test')

    def test_write_to_folder(self):
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')

        test_new_model_name = 'model1'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_new_model_name = 'model2'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [1, 0, 1],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')

        self.assertCountEqual(['model1', 'model2'], test_sklearn_batch.model_dict.keys())

        test_sklearn_batch.write_to_folder()

        self.assertTrue(os.path.isdir('test'))
        self.assertTrue(os.path.isfile('test/column_names.pkl'))
        self.assertTrue(os.path.isfile('test/model1.pkl'))
        self.assertTrue(os.path.isfile('test/model2.pkl'))

        # Delete created folder
        shutil.rmtree('test')

    def test_predict(self):
        '''
            Test that models can be trained and predicted with.
        '''
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')

        test_new_model_name = 'model1'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')
        test_new_model_name = 'model2'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [1, 0, 1],
            'text_column': ['data1', 'data2', 'data3']
        })
        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')

        self.assertCountEqual(['model1', 'model2'], test_sklearn_batch.model_dict.keys())

        test_examples = pd.DataFrame({
            'feature1': [5, 6, 7],
            'feature2': [6, 12, 18],
            'text_column': ['data1', 'data2', 'data3']
        })
        expected_label_predictions = pd.DataFrame({
            'text_column': ['data1', 'data2', 'data3'],
            'model1': [-8.4364947759189, -8.4364947759189, -8.4364947759189],
            'model2': [8.4364947759189, 8.4364947759189, 8.4364947759189]
        })
        actual_label_predictions = test_sklearn_batch.predict(test_examples, 'text_column')

        pd.testing.assert_frame_equal(expected_label_predictions, actual_label_predictions, check_like=True)

    def test_train_new(self):
        '''
            Verify that a new model can be trained when valid data is supplied.
        '''
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')

        test_new_model_name = 'new_model'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })

        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')

        self.assertIn('new_model', test_sklearn_batch.model_dict.keys())

    def test_train_new_unoriginal_model_name(self):
        '''
            Verify behavior when attempting to train a new model whose name
            already exists.
        '''
        test_sklearn_batch = SklearnBatch(name='test', feature_column_names=['feature1', 'feature2'], label_column_name='label')

        test_new_model_name = 'new_model'
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6],
            'label': [0, 1, 0],
            'text_column': ['data1', 'data2', 'data3']
        })

        test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')

        self.assertIn('new_model', test_sklearn_batch.model_dict.keys())

        with self.assertRaises(ValueError):
            test_sklearn_batch.train_new(test_new_model_name, test_data, 'text_column')

if __name__ == "__main__":
    unittest.main()
