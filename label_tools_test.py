from label_tools import *
import unittest
from unittest.mock import patch
from io import StringIO

class TestLabelTools(unittest.TestCase):

    @patch("sys.stdin", StringIO('\n'.join(['1', '0', '1', '0', '1'])))
    def test_label_data_ordered(self):
        """
            Verify that data can be labelled when shown in its entriety in order (non-shuffled).
        """
        data_to_be_labeled = list(range(5))
        expected_post_labeled_data = pd.DataFrame({
            'data': list(range(5)),
            'response': ['1', '0', '1', '0', '1']
        })
        actual_post_labeled_data = label_data(data_to_be_labeled, shuffle=False, prompt_user=False)
        pd.testing.assert_frame_equal(expected_post_labeled_data, actual_post_labeled_data)

    @patch("sys.stdin", StringIO('\n'.join(['1', '0', '1', '0', '1'])))
    def test_label_data_randomly_ordered(self):
        """
            Verify that data can be labelled when shown in its entriety in a random order (shuffled).
        """
        data_to_be_labeled = list(range(5))
        post_labeled_data = label_data(data_to_be_labeled, shuffle=True, prompt_user=False)

        # Verify that data is labelled with expected responses, and that all elements are present (though not necessarily in the original order)
        self.assertEqual(list(post_labeled_data['response']), ['1', '0', '1', '0', '1'])
        self.assertCountEqual(data_to_be_labeled, post_labeled_data['data'])

    @patch("sys.stdin", StringIO('\n' * 5))
    def test_label_data_default_response(self):
        """
            Verify that data can be correctly labelled when shown in its entriety, and the user supplies the empty response.
        """
        data_to_be_labeled = list(range(5))
        expected_labeled_data = pd.DataFrame({
            'data': list(range(5)),
            'response': ['0', '0', '0', '0', '0']
        })
        actual_labeled_data = label_data(data_to_be_labeled, shuffle=False, label_responses={'0', '1', ''}, default_response='0', prompt_user=False)
        pd.testing.assert_frame_equal(expected_labeled_data, actual_labeled_data)

    @patch("sys.stdin", StringIO('stop\n'))
    def test_label_data_no_response(self):
        """
            Verify that having a user stop the labelling process without any responses behaves as is ecpected.
        """
        data_to_be_labeled = list(range(5))
        self.assertRaises(ValueError, label_data, data_to_be_labeled, shuffle=False, stop_response='stop', prompt_user=False)

    @patch("sys.stdin", StringIO('\n'.join(['1', '0', '1', 'stop'])))
    def test_label_data_early_stopping(self):
        """
            Verify that data can be partially labelled.
        """
        data_to_be_labeled = list(range(5))
        expected_post_labeled_data = pd.DataFrame({
            'data': list(range(3)),
            'response': ['1', '0', '1']
        })
        actual_post_labeled_data = label_data(data_to_be_labeled, shuffle=False, prompt_user=False)
        pd.testing.assert_frame_equal(expected_post_labeled_data, actual_post_labeled_data)

if __name__ == "__main__":
    unittest.main()
