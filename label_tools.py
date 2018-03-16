"""
    Tools that assist with obtaining labels from user via IO.
"""

import pandas as pd
import numpy as np

def label_data(elements, response_column_name='response', shuffle=True, label_responses={'0', '1'}, default_response='0', stop_response='stop', prompt_user=True):
    """
        Obtain labels for the data.
        Data is shown to user randomly by default.
        Valid responses and default (response when nothing is given) response can be specified.
        Stop response can be specified, and will halt the labelling process and return only what was labelled.
        Responses will be stored in a specified column with all data that was initially present in the dataframe.
        prompt_user can be set to False for testing, and will not cue user for responses.
    """

    dataframe = pd.DataFrame({
        'data': elements
    })

    if shuffle:
        dataframe = dataframe.sample(frac=1)

    possible_responses = label_responses | {''} | {stop_response}

    responses = []
    row_indices_responded_to = []
    for index, row in dataframe.iterrows():

        if prompt_user:
            print('\n' * 100)
            print(row['data'])
            print('Enter your response')
        response = input()

        while response not in possible_responses:
            if prompt_user:
                print('\n' * 100)
                print(row['data'])
            print('Unrecognized response. Please try again')
            response = input()
        if response == stop_response:
            break
        if response == '':
            response = default_response

        responses.append(response)
        row_indices_responded_to.append(index)

    if len(responses) == 0:
        raise ValueError('User has not responded to any rows')

    responed_dataframe = dataframe.loc[row_indices_responded_to, :]
    responed_dataframe[response_column_name] = responses
    return responed_dataframe
