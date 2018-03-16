import unittest
from stat_tools import *
import pandas as pd
import numpy as np

# TODO:
# - Note that some testing methods are probabalistic (small chance of false negative)
class StatToolsTest(unittest.TestCase):

    def test_apply_temperature_regular(self):
        """
            Verify that regular temperatures can be applied to regular probabilities.
        """

        # Scale to be applied to probabilities
        test_temperature = .8

        # Original probabilities
        test_probabilities = np.array([.2, .2, .6])

        expected_scaled_probabilities = np.array([ 0.168117,  0.168117,  0.663765])
        actual_scaled_probabilities = apply_temperature(test_probabilities, test_temperature)
        np.testing.assert_allclose(expected_scaled_probabilities, actual_scaled_probabilities, rtol=1e-03)

        # Repeat with smaller temperature (makes probabilities LESS uniform)
        test_temperature = .2
        test_probabilities = np.array([.2, .2, .6])
        expected_scaled_probabilities = np.array([ 0.004082,  0.004082,  0.991837])
        actual_scaled_probabilities = apply_temperature(test_probabilities, test_temperature)
        np.testing.assert_allclose(expected_scaled_probabilities, actual_scaled_probabilities, rtol=1e-03)

        # Repeat with larger temperature (makes probabilities MORE uniform)
        test_temperature = 1.5
        test_probabilities = np.array([.2, .2, .6])
        expected_scaled_probabilities = np.array([ 0.245093,  0.245093,  0.509814])
        actual_scaled_probabilities = apply_temperature(test_probabilities, test_temperature)
        np.testing.assert_allclose(expected_scaled_probabilities, actual_scaled_probabilities, rtol=1e-03)

    def test_apply_temperature_of_one(self):
        """
            Verify that temperature of 1 has no effect on probabilities.
        """
        for probabilities in [np.array([.1, .1, .8]), np.array([.5, .5]), np.array([1, 0])]:
            np.testing.assert_array_equal(apply_temperature(probabilities, 1.0), probabilities)

    def test_apply_temperature_extremes(self):
        """
            Verify behavior when extreme values fed as temperature.
        """
        test_temperature = .05
        test_probabilities = np.array([.2, .2, .6])
        expected_scaled_probabilities = np.array([ 0.0,  0.0,  1])
        actual_scaled_probabilities = apply_temperature(test_probabilities, test_temperature)
        np.testing.assert_allclose(expected_scaled_probabilities, actual_scaled_probabilities, rtol=1e-03)

        test_temperature = 1000
        test_probabilities = np.array([.2, .2, .6])
        expected_scaled_probabilities = np.array([ 1/3,  1/3,  1/3])
        actual_scaled_probabilities = apply_temperature(test_probabilities, test_temperature)
        np.testing.assert_allclose(expected_scaled_probabilities, actual_scaled_probabilities, rtol=1e-03)

    def test_apply_temperature_invalid_temperature(self):
        """
            Verify behavior when invalid values fed as temperature smaller than .05)
        """
        test_temperature = .04
        test_probabilities = np.array([.2, .2, .6])
        self.assertRaises(ValueError, apply_temperature, test_probabilities, test_temperature)

        test_temperature = 0
        test_probabilities = np.array([.2, .2, .6])
        self.assertRaises(ValueError, apply_temperature, test_probabilities, test_temperature)

        test_temperature = -1
        test_probabilities = np.array([.2, .2, .6])
        self.assertRaises(ValueError, apply_temperature, test_probabilities, test_temperature)

    def test_apply_temperature_probability_sum(self):
        """
            Verify that applying temperature to probabilities never produces
            probabilities that do not sum to 1.
        """
        test_probabilities = np.array([0, .1, .8, .1])
        for i in range(1, 101):
            temperature = i / 10
            scaled_probabilities = apply_temperature(test_probabilities, temperature)
            self.assertAlmostEqual(sum(scaled_probabilities), 1, places=5)

        test_probabilities = np.array([.5, .5])
        for i in range(1, 101):
            temperature = i / 10
            scaled_probabilities = apply_temperature(test_probabilities, temperature)
            self.assertAlmostEqual(sum(scaled_probabilities), 1, places=5)

        test_probabilities = np.array([1/3, 1/3, 1/3])
        for i in range(1, 101):
            temperature = i / 10
            scaled_probabilities = apply_temperature(test_probabilities, temperature)
            self.assertAlmostEqual(sum(scaled_probabilities), 1, places=5)

    def setUp(self):
        """
            Create mock distribution.
        """

        # Lower level dataframes
        lower_case = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.25, .25, .5]
        })
        capitals = pd.DataFrame({
            'elements': ['A', 'B', 'C'],
            'probabilities': [.25, .25, .5]
        })
        teens = pd.DataFrame({
            'elements': ['13', '14', '15'],
            'probabilities': [.25, .25, .5]
        })
        hundreds = pd.DataFrame({
            'elements': ['103', '104', '105'],
            'probabilities': [.25, .25, .5]
        })

        # Middle level dataframes
        letters = pd.DataFrame({
            'elements': [lower_case, capitals],
            'probabilities': [.5, .5]
        })
        numbers = pd.DataFrame({
            'elements': [teens, hundreds],
            'probabilities': [.5, .5]
        })

        # Top level dataframe
        items = pd.DataFrame({
            'elements': [letters, numbers],
            'probabilities': [.5, .5]
        })

        self.test_distribution = items

    def test_element_distribution(self):
        test_elements = [1, 1, 2, 1]
        expected_distribution = pd.DataFrame({
            'elements': [1, 2],
            'probabilities': [3/4, 1/4]
        })
        actual_distribution = element_distribution(test_elements)
        pd.testing.assert_frame_equal(expected_distribution, actual_distribution)

        test_elements = ['b', 'c', 'a', 'a']
        expected_distribution = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': np.array([1/2, 1/4, 1/4])
        })
        actual_distribution = element_distribution(test_elements)
        pd.testing.assert_frame_equal(expected_distribution, actual_distribution)

    def test_element_distribution_non_hashables(self):
        """
            Verify that un-hashable elements are handled correctly.
        """
        test_elements = [[1], [1, 2]]
        self.assertRaises(TypeError, element_distribution, test_elements)

    def test_avg_element_frequency(self):
        """
            Verify that average frequency can be calculated for elements.
        """
        self.assertAlmostEqual(avg_element_frequency([1, 1, 2, 1]), 2)
        self.assertAlmostEqual(avg_element_frequency([1, 2, 3, 4]), 1)
        self.assertAlmostEqual(avg_element_frequency([1, 2, 2, 3]), 1.3333, places=2)
        self.assertAlmostEqual(avg_element_frequency(['a', 'a', 'b', 'a']), 2)
        self.assertAlmostEqual(avg_element_frequency(['a', 'b', 'c', 'd']), 1)
        self.assertAlmostEqual(avg_element_frequency(['a', 'b', 'b', 'c']), 1.3333, places=2)

    def test_avg_element_frequency_non_hashables(self):
        """
            Verify that un-hashable elements are handled correctly.
        """
        test_elements = [[1], [1, 2]]
        self.assertRaises(TypeError, avg_element_frequency, test_elements)

    def test_sample_items_contained(self):
        """
            Verify that sampling from a distribution produces only items contained in the distribution.

            Testing methods assumes that element_distribution works properly.
        """
        test_distribution = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.25, .25, .5]
        })
        sampled_items = sample(test_distribution, 'elements', n=1000000, probability_column_name='probabilities')
        self.assertTrue(set(test_distribution['elements']) >= set(sampled_items))

    def test_sample_element_distribution_inverse(self):
        """
            Verify that producing a distribution from data and sampling from
            a distribution act like inverse of each other when many samples are
            drawn.
        """
        test_distribution = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.25, .25, .5]
        })

        pd.testing.assert_frame_equal(element_distribution(sample(test_distribution, 'elements', n=1000000, probability_column_name='probabilities')), test_distribution, check_less_precise=2)


    def test_sample_temperature(self):
        """
            Verify that distribution can be sampled from after having temperature
            applied to probabilities.
        """
        test_distribution = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.25, .25, .5]
        })
        expected_sampled_distribution = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [0.278367, 0.279475, 0.442158]
        })
        sampled_elements = sample(test_distribution, 'elements', n=1000000, probability_column_name='probabilities', temperature=1.5)
        actual_sampled_distribution = element_distribution(sampled_elements)

        pd.testing.assert_frame_equal(expected_sampled_distribution, actual_sampled_distribution, check_less_precise=2)

    def test_sample_probabilities_inferred(self):
        """
            Verify that sample can infer uniform probabilities when they are not
            provided.
        """
        test_distribution = pd.DataFrame({
            'elements': ['a', 'b', 'c']
        })
        expected_sampled_distribution = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [1/3, 1/3, 1/3]
        })
        sampled_elements = sample(test_distribution, 'elements', n=1000000)
        actual_sampled_distribution = element_distribution(sampled_elements)

        pd.testing.assert_frame_equal(expected_sampled_distribution, actual_sampled_distribution, check_less_precise=2)

    def test_combine_distributions_given_both_probabilities(self):
        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2, 3],
            'probabilities': [.2, .2, .6]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.4, .4, .2]
        })
        test_combine_probabilities = [.5, .5]
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 3, 'a', 'b', 'c'],
            'probabilities': [.1, .1, .3, .2, .2, .1]
        })
        actual_combined_distribution = combine_distributions(
            distributions=[test_distribution_1, test_distribution_2],
            distribution_probability_column_name='probabilities',
            combined_probabilities=test_combine_probabilities
        )
        pd.testing.assert_frame_equal(expected_combined_distrubition, actual_combined_distribution)

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2, 3],
            'probabilities': [.2, .2, .6]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.4, .4, .2]
        })
        test_combine_probabilities = [.25, .75]
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 3, 'a', 'b', 'c'],
            'probabilities': [.05, .05, .15, .3, .3, .15]
        })
        actual_combined_distribution = combine_distributions(
            distributions=[test_distribution_1, test_distribution_2],
            distribution_probability_column_name='probabilities',
            combined_probabilities=test_combine_probabilities
        )
        pd.testing.assert_frame_equal(expected_combined_distrubition, actual_combined_distribution)

    def test_combine_distributions_infered_uniform_outer_probabilities(self):

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2, 3],
            'probabilities': [.2, .2, .6]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.4, .4, .2]
        })
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 3, 'a', 'b', 'c'],
            'probabilities': [.1, .1, .3, .2, .2, .1]
        })
        actual_combined_distribution = combine_distributions(
            distributions=[test_distribution_1, test_distribution_2],
            distribution_probability_column_name='probabilities',
            infer_probabilities='uniform'
        )
        pd.testing.assert_frame_equal(expected_combined_distrubition, actual_combined_distribution)

    def test_combine_distributions_infered_quantity_outer_probabilities(self):

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2],
            'probabilities': [.4, .6]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c', 'd'],
            'probabilities': [.4, .4, .1, .1]
        })
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 'a', 'b', 'c', 'd'],
            'probabilities': [.4 / 3, .2, .8 / 3, .8 / 3, .2 / 3, .2 / 3]
        })
        actual_combined_distribution = combine_distributions(
            distributions=[test_distribution_1, test_distribution_2],
            distribution_probability_column_name='probabilities',
            infer_probabilities='quantity'
        )
        pd.testing.assert_frame_equal(expected_combined_distrubition, actual_combined_distribution)

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2, 3],
            'probabilities': [.2, .2, .6]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.4, .4, .2]
        })
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 3, 'a', 'b', 'c'],
            'probabilities': [.1, .1, .3, .2, .2, .1]
        })
        actual_combined_distribution = combine_distributions(
            distributions=[test_distribution_1, test_distribution_2],
            distribution_probability_column_name='probabilities',
            infer_probabilities='quantity'
        )
        pd.testing.assert_frame_equal(expected_combined_distrubition, actual_combined_distribution)

    def test_combine_distributions_infered_inner_probabilities(self):

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c', 'd']
        })
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 'a', 'b', 'c', 'd'],
            'probabilities': [.25, .25, .125, .125, .125, .125]
        })
        actual_combined_distribution = combine_distributions(
            distributions=[test_distribution_1, test_distribution_2],
            infer_probabilities='uniform'
        )
        pd.testing.assert_frame_equal(expected_combined_distrubition, actual_combined_distribution)

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c', 'd']
        })
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 'a', 'b', 'c', 'd'],
            'probabilities': [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, ]
        })
        actual_combined_distribution = combine_distributions(
            distributions=[test_distribution_1, test_distribution_2],
            infer_probabilities='quantity'
        )
        pd.testing.assert_frame_equal(expected_combined_distrubition, actual_combined_distribution)

    def test_combine_distributions_invalid_outer_probabilities(self):

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2, 3],
            'probabilities': [.2, .2, .6]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.4, .4, .2]
        })
        test_combine_probabilities = [.5, .6]
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 3, 'a', 'b', 'c'],
            'probabilities': [.1, .1, .3, .2, .2, .1]
        })
        self.assertRaises(ValueError, combine_distributions,
            distributions=[test_distribution_1, test_distribution_2],
            distribution_probability_column_name='probabilities',
            combined_probabilities=test_combine_probabilities
        )

        test_distribution_1 = pd.DataFrame({
            'elements': [1, 2, 3],
            'probabilities': [.2, .2, .6]
        })
        test_distribution_2 = pd.DataFrame({
            'elements': ['a', 'b', 'c'],
            'probabilities': [.4, .4, .2]
        })
        test_combine_probabilities = [.5, .4]
        expected_combined_distrubition = pd.DataFrame({
            'elements': [1, 2, 3, 'a', 'b', 'c'],
            'probabilities': [.1, .1, .3, .2, .2, .1]
        })
        self.assertRaises(ValueError, combine_distributions,
            distributions=[test_distribution_1, test_distribution_2],
            distribution_probability_column_name='probabilities',
            combined_probabilities=test_combine_probabilities
        )

if __name__ == "__main__":
    unittest.main()
