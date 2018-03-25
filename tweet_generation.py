import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from math import isclose
from textgenrnn.textgenrnn import textgenrnn_encode_sequence
from stat_tools import *

from textgenrnn import textgenrnn
from parse_tools import ParseTools
import re
import time

# NOTE:
# - Custom generation was done in response to generated tweets being too long, having too few punctuations, etc.

def custom_generate(textgenrnn_model, prefix=None, temperature=0.2,
                    max_gen_length=200, weight_adjustments={}, include_stop_token=False):

    # Obtain parameters from textgenrnn object
    model = textgenrnn_model.model
    vocab = textgenrnn_model.vocab
    indices_char = textgenrnn_model.indices_char

    # Parameters in which all models are trained with
    maxlen = 40
    meta_token = '<s>'

    text = [meta_token] + list(prefix) if prefix else [meta_token]
    next_char = ''

    while next_char != meta_token and len(text) < max_gen_length:
        encoded_text = textgenrnn_encode_sequence(text[-maxlen:], vocab, maxlen)

        # Ignore probability of first element (not accounted for in index char)
        preds = model.predict(encoded_text, batch_size=1)[0][1:]

        # Perform weight scaling where applicable
        # +1 accounts for unused 0-key
        for index, value in enumerate(preds):
            if indices_char[index + 1] in weight_adjustments:
                preds[index] = value * weight_adjustments[indices_char[index + 1]]

        # Scale back to 1
        preds /= np.sum(preds)

        # Apply temperature scaling
        preds = apply_temperature(preds, temperature)

        # Sample from probabilities to select next characters
        next_index = np.argmax(np.random.multinomial(1, preds, 1))
        next_char = indices_char[next_index + 1]  # +1 accounts for unused 0-key
        text += [next_char]
    if include_stop_token:
        return ''.join(text[1:])
    else:
        return ''.join(text[1:-1])

def generate(model, gen_count, temperature, weight_adjust={'.': 2, '?': 2, '!': 2, ',': 2}):

    generations = []

    for _ in range(gen_count):

        generation = custom_generate(model, temperature=temperature, weight_adjustments=weight_adjust, include_stop_token=True)

        # If generation was able to finish (as opposed to being manually stopped),
        # and if tweet contains letters (is not just series of symbols)
        if generation[-3:] == '<s>' and ParseTools.contains_letters(generation[:-3]):

            # Clip stop character and save tweet
            generation = generation[:-3]
            generations.append(generation)

    return generations
