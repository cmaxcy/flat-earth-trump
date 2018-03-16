import pandas as pd
from parse_tools import *

def apply_cleans(tweet, cleans):
    for clean in cleans:
        tweet = clean(tweet)
    return tweet

def clean_tweets(tweets):
    ats = set(pd.read_pickle('Data/ats')['ats'])
    cleans = [lambda x: fix_ats(x, ats), re_apostrophize, re_amp, reduce_punctuations]
    cleaner = lambda x: apply_cleans(x, cleans)
    return list(map(cleaner, tweets))
