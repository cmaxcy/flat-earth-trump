from math import isclose
import numpy as np
import pandas as pd
from collections import Counter
import random

def element_distribution(elements):
    """
        Returns pandas DataFrame mapping elements to the probability
        of their occurence.
    """
    counter = Counter(elements)
    total_elements = len(elements)

    unique_elements = []
    unique_element_probabilities = []
    for unique_element in sorted(counter):
        unique_elements.append(unique_element)
        unique_element_probabilities.append(counter[unique_element] / total_elements)

    return pd.DataFrame({
        'elements': unique_elements,
        'probabilities': unique_element_probabilities
    })

# TODO:
# - Test for 0 element case
def avg_element_frequency(elements):
    """
        Return the average frequency of unique elements.
        Elements must be hashable.
    """

    if len(elements) == 0:
        return 0

    counter = Counter(elements)
    return sum(counter.values()) / len(counter.keys())

# TODO:
# - Consider NaN element case
def sample(data, element_column_name, n=1, probability_column_name=None, temperature=1.0):

    if probability_column_name is None:
        weights = None
        # TODO: consider informing user when temperature is attempted to be applied here
    else:
        weights = np.array(data[probability_column_name])
        weights = apply_temperature(weights, temperature)

    return list(data.sample(n=n, weights=weights, replace=True)[element_column_name])

def combine_distributions(distributions, distribution_probability_column_name=None, combined_probabilities=None, infer_probabilities='uniform'):

    # Add probability column to distributions if one is not specified
    # Uniform probabilities will be used
    if distribution_probability_column_name is None:
        distribution_probability_column_name = 'probabilities'
        for distribution in distributions:
            distribution_element_count = distribution.shape[0]
            distribution[distribution_probability_column_name] = np.full(distribution_element_count, 1 / distribution_element_count)

    # Infer combination probabilities if not passed. Uniform gives every
    # distribution an equal chance of appearing. quantity scales the
    # probabilities of the distributions with the number of elements in them
    # compared to the total elements.
    if combined_probabilities is None:
        if infer_probabilities is 'uniform':
            combined_probabilities = [1 / len(distributions) for _ in distributions]
        elif infer_probabilities is 'quantity':
            elements_in_distributions = sum(distribution.shape[0] for distribution in distributions)
            combined_probabilities = [distribution.shape[0] / elements_in_distributions for distribution in distributions]
        else:
            raise ValueError('Invalid inference scheme passed')

    if len(distributions) != len(combined_probabilities):
        raise ValueError('Distributions and combined_probabilities are not consistent in size')

    # Ensure usable probabilities are passed
    if not isclose(sum(combined_probabilities), 1):
        raise ValueError("Combine probabilities do not sum to 1")

    # Scale distributions by probabilities
    scaled_distributions = []
    for probability, distribution in zip(combined_probabilities, distributions):
        distribution_probabilities = np.array(distribution[distribution_probability_column_name])

        # Ensure usable probabilities are contained in each distribution
        if not isclose(sum(distribution_probabilities), 1):
            raise ValueError("Distribution probabilities do not sum to 1")

        distribution[distribution_probability_column_name] = probability * distribution_probabilities
        scaled_distributions.append(distribution)

    # Stack scaled distributions
    return pd.concat(scaled_distributions, ignore_index=True)

# TODO:
# - Provide link to sources
def apply_temperature(preds, temperature):
    """
        Apply temperature scaling to predictions.
    """

    # Temperature of 1 will have no effect on predictions
    if temperature == 1.0:
        return preds

    # Catch invalid temperature
    if temperature < .05:
        raise ValueError('Temperature is too small to avoid arithmetic errors')

    # Perform temperature scaling
    preds = np.log(preds + 1e-12) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # Reduce risk of probabilities exceeding 1.0
    preds = preds - 1e-06
    return preds.clip(min=0)
