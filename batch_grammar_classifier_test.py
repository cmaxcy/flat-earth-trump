import unittest
import pandas as pd
from batch_grammar_classifier import BatchGrammarClassifier
import shutil
import os

# TODO: ensure prediction maintains order

class TestBatchGrammarClassifier(unittest.TestCase):

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

        actual_output = BatchGrammarClassifier.label_apply(test_data, test_functions)

        pd.testing.assert_frame_equal(expected_output, actual_output)

    def test_constructor(self):
        test_batch_grammar_classifier = BatchGrammarClassifier('test')

    def test_load_from_folder_non_existant_folder(self):
        '''
            Verify that a non-existant folder can be caught.
        '''
        with self.assertRaises(ValueError):
            BatchGrammarClassifier.load_from_folder('not_a_folder')

    def test_format_examples(self):
        '''
            Verify that example formatting is performed correctly
        '''
        test_negative_examples = pd.DataFrame({
            'example': ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        })
        test_positive_examples = pd.DataFrame({
            'example': ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']
        })

        expected_formatted_examples = pd.DataFrame({
            'example': ['Horrible here grammar.', 'Nasty one', 'Great grammar here!', 'Another great one'],
            'label': [0, 0, 1, 1]
        })
        actual_formatted_examples = BatchGrammarClassifier.format_examples(test_negative_examples, test_positive_examples)

    def test_train_new(self):
        '''
            Verify that a new model can be trained when valid data is supplied.
        '''
        test_batch_grammar_classifier = BatchGrammarClassifier(name='test')

        test_new_model_name = 'new_model'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        self.assertIn('new_model', test_batch_grammar_classifier.model_dict.keys())

    def test_train_new_unoriginal_model_name(self):
        '''
            Verify behavior when attempting to train a new model whose name
            already exists.
        '''
        test_batch_grammar_classifier = BatchGrammarClassifier(name='test')

        test_new_model_name = 'new_model'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        self.assertIn('new_model', test_batch_grammar_classifier.model_dict.keys())

        with self.assertRaises(ValueError):
            test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

    def test_train_new_models_independent(self):
        '''
            Verify that when training a new model, the model is different than
            any existing ones.
        '''
        test_batch_grammar_classifier = BatchGrammarClassifier(name='test')

        test_new_model_name = 'model1'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        test_new_model_name = 'model2'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        model1 = test_batch_grammar_classifier.model_dict['model1']
        model2 = test_batch_grammar_classifier.model_dict['model2']

        self.assertNotEqual(model1, model2)

    def test_predictions_consistent_after_load(self):
        '''
            Verify that model output is unchanged after instance is read and
            written to file.
        '''
        test_name = 'test'
        test_batch_grammar_classifier = BatchGrammarClassifier(name=test_name)

        test_new_model_name = 'model1'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        test_new_model_name = 'model2'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        test_examples = ['Here are two new examples', 'How grammatically correct are they?']

        # Predictions before folder write
        test_pre_write_predictions = test_batch_grammar_classifier.predict(test_examples)

        # Write then retrive instance
        test_batch_grammar_classifier.write_to_folder()
        test_read_instance = BatchGrammarClassifier.load_from_folder(test_name)
        self.assertEqual(test_read_instance.name, test_name)

        # Predictions after folder write
        test_post_write_predictions = test_read_instance.predict(test_examples)
        pd.testing.assert_frame_equal(test_pre_write_predictions, test_post_write_predictions)

        # Delete created folder
        shutil.rmtree(test_name)

    def test_load_from_folder_non_existant_folder(self):
        '''
            Verify that a non-existant folder can be caught.
        '''
        with self.assertRaises(ValueError):
            BatchGrammarClassifier.load_from_folder('not_a_folder')

    def test_write_no_clobber(self):

        # Build and write instance to folder
        test_name = 'test'
        test_batch_grammar_classifier = BatchGrammarClassifier(name=test_name)

        test_new_model_name = 'model1'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        test_new_model_name = 'model2'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        test_batch_grammar_classifier.write_to_folder()
        self.assertTrue(os.path.isdir('test'))
        self.assertTrue(os.path.isfile('test/model1.pkl'))
        self.assertTrue(os.path.isfile('test/model2.pkl'))

        with self.assertRaises(ValueError):
            test_batch_grammar_classifier.write_to_folder(clobber=False)

        # Delete created folder
        shutil.rmtree('test')

    def test_write_clobber(self):

        # Build and write instance to folder
        test_name = 'test'
        test_batch_grammar_classifier = BatchGrammarClassifier(name=test_name)

        test_new_model_name = 'model1'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        test_new_model_name = 'model2'
        test_negative_examples = ['Horrible here grammar.', 'Nasty one', 'Does grammar work not', 'still work no']
        test_positive_examples = ['Great grammar here!', 'Another great one', 'Spectacular grammar here', 'Exellent']

        test_batch_grammar_classifier.train_new(test_new_model_name, test_negative_examples, test_positive_examples)

        test_batch_grammar_classifier.write_to_folder()
        self.assertTrue(os.path.isdir('test'))
        self.assertTrue(os.path.isfile('test/model1.pkl'))
        self.assertTrue(os.path.isfile('test/model2.pkl'))

        test_batch_grammar_classifier.write_to_folder()
        self.assertTrue(os.path.isdir('test'))
        self.assertTrue(os.path.isfile('test/model1.pkl'))
        self.assertTrue(os.path.isfile('test/model2.pkl'))

        # Delete created folder
        shutil.rmtree('test')

if __name__ == "__main__":
    unittest.main()
