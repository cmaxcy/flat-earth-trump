from label_tools import *
from tweet_tool import *
import time
from random import shuffle
import pandas as pd
from parse_tools import *
from tweet_generation import *
from textgenrnn import textgenrnn
from tweet_clean import *
from batch_grammar_classifier import BatchGrammarClassifier

# Generate tweets
model = textgenrnn('textgenrnn_FET_model')
generated_tweets = generate(model, gen_count=100, temperature=.6)

# Apply simple fixes to tweets (@ correction, apostrophe insertion, etc.)
cleaned_tweets = clean_tweets(generated_tweets)

# Sort tweets by calculated grammar score
bgc = BatchGrammarClassifier.load_from_folder('grammar_models')
ranked_tweets = bgc.rank(cleaned_tweets)

# Obtain hand labels
labelled_tweets = label_data(ranked_tweets, shuffle=False)
good_tweets = list(labelled_tweets.loc[labelled_tweets['response'] == '1']['data'])

# Schedule tweets
shuffle(good_tweets)
t = TweetTool()
times = t.get_times(len(good_tweets), 100, 100)
t.schedule_tweets(good_tweets, times)
