import re
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
from collections import Counter
import random
import string
from stat_tools import *

# TODO:
# - Include links to libary methods
# - Test and document all unintended side effects of methods (for example
#   when combine_words is used and extraneous spaces are removed)
# - Consider ordering methods, and matching tests with order
# - Look into using word vectors and performing summation

# Regular expressions for operating on tweets
twitter_name_re = r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)'
twitter_pic_link_re = r"pic.twitter\S+"
twitter_link_re = r"http\S+"
twitter_ht_re = r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9]+)'
apostrophe_re = r"[\w]+'[\w]+"
triple_period_re = r"\.\.\.(\.)*"

def reduce_punctuations(tweet):
    '''
        Replace sequences of over 3 punctuations with a smaller sequence.
    '''
    for punctuation in {'!', '?', '.'}:
        for sequence in length_n_sequences(tweet, punctuation, 3):
            replacement = punctuation * (abs(int(np.random.normal())) + 1)
            tweet = tweet.replace(sequence, replacement, 1)

    return tweet

def fix_ats(tweet, at_set):
    """
        Replace all ats in tweet with closest match in at_set
    """
    at_map = dict()
    tweet_ats = extract_ats(tweet)
    for tweet_at in tweet_ats:
        at_map[tweet_at] = find_nearest_string(tweet_at, at_set)

    for tweet_at in tweet_ats:
        tweet = tweet.replace(tweet_at, at_map[tweet_at])

    return tweet

def contains_letters(phrase):
    return set(string.ascii_lowercase).intersection(phrase.lower()) != set()

def get_count_funcs(words):
    def finds(word):
        func = lambda x: len(re.findall(word, x))
        func.__name__ = 'count-' + word
        return func
    return [finds(i) for i in words]

def avg_word_frequency(string):
    return avg_element_frequency(extract_words(string))

def count_ats(string):
    return len(extract_ats(string))

def count_at_prefixes(string):
    return len(extract_ats(string)) - len(extract_ats(remove_at_prefixes(string)))

def re_apostrophize(string):
    return re.sub(' s ', '\'s ', string)

def re_amp(string):
    return re.sub('&amp;', '&', string)

def length_n_sequences(string, char, n):
    '''
        Finds all characters sequences of length >= n in the string.
    '''
    char = re.escape(char)
    regex = char * n + char + '*'
    return re.compile(regex).findall(string)

def is_proper_sentence(input_string):
    """
        Returns whether or not the sentence starts with a capital letter and ends with a punctuation.
    """
    if input_string == '':
        return False

    punctuations = {'.', '?', '!'}
    uppercase_letters = set(string.ascii_uppercase)

    if input_string[0] in uppercase_letters and input_string[-1] in punctuations:
        return True

    return False

def replace_all(keeps, replacement, string):
    replacement_set = set(string) - set(keeps)
    for char in replacement_set:
        string = re.sub(char, replacement, string)
    return string

def split_join(string):
    """
        Ensure that all words are separated only by a single space.
    """
    return ' '.join(string.split())

def punctuate(string):
    """
        Add punctuation to end of string if not already present. Space-like characters (tabs, carriage returns, etc.) are removed as well.
        If string ends in character, last word will be dropped.
    """
    punctuations = ['.', '?', '!']

    if string in punctuations:
        return string

    # Empty and single space string cases
    if len(string) <= 1:
        return random.choice(punctuations)

    # Remove last word if string ends in character
    if string[-1].isalpha():
        string = ' '.join(string.split()[:-1])
    else:
        string = split_join(string)

    # String may have had all contents removed
    if len(string) == 0:
        return random.choice(punctuations)

    if string[-1] not in punctuations:
        string += random.choice(punctuations)

    return string

def extract_apostrophe_words(string):
    return re.compile(apostrophe_re).findall(string)

def replace_ats(string, replacement):
    """
        Replace all twitter names.
    """
    return re.sub(twitter_name_re, replacement, string)

def is_quoted_tweet(string):
    """
        Returns whether or not the string passed is a quoted tweet.

        Quoted tweets are as follows:

        @VeryOddDog: What\'s BRUTAL is a nation WITHOUT Trump!

        Surrounded by quotes.
    """

    # Null strings and characters are not quoted tweets
    if len(string) <= 1:
        return False

    if string[0] == '\'' or string[0] == '\"':
        quote = string[0]
        if string[1] == '@':
            at_less_string = remove_ats(string)
            if at_less_string[1] == ':':
                return True

    return False

def remove_outer_quotes(string):
    """
        Removes outer quotes from the string. If outer quotes are not present, string is return unaltered.
    """

    if len(string) <= 1:
        return string

    if string[0] == '"' and string[-1] == '"':
        return string[1:-1]
    elif string[0] == '\'' and string[-1] == '\'':
        return string[1:-1]
    else:
        return string

def extract_sentences(string):
    """
        Returns senteces from string. Sentences need not be grammatically
        correct.
    """
    return sent_tokenize(string)

def extract_words(string, twitter_words=True):
    """
        Return words from string. Twitter words (handles, links, hashtags)
        are kept intact by default.
    """

    # Compile set of hard to extract words
    placeholder_mappings = dict()
    words_to_map = set(extract_apostrophe_words(string))
    if twitter_words is True:
        words_to_map |= set(extract_ats(string) + extract_http_links(string) + extract_pic_links(string) + extract_hts(string))

    # Map all hard to extract words to unique random strings
    for word in words_to_map:
        random_string = get_random_string(16)
        placeholder_mappings[random_string] = word
        string = string.replace(word, random_string)

    # Remove symbols, splti on spaces, and replace hard to extract placeholders with their original words
    word_list = re.sub(r'[^\w]', ' ', string).split()
    return list(map(lambda x: placeholder_mappings[x] if x in placeholder_mappings else x, word_list))

def remove_dots(string):
    return re.sub("…", "", string)

def extract_ats(string):
    """
        Returns every twitter handle (ie. @realDonaldTrump) in the order in
        which they occur in the string.
    """

    # Replace all '.@' occurences with '@'
    string = re.sub('\.@', '@', string)

    ats = []
    for at in re.findall(twitter_name_re, string):
        ats.append('@' + at)
    return ats

def remove_ats(string):
    """
        Removes all twitter handles (ie. @realDonaldTrump) from the string.
    """
    return re.sub(twitter_name_re, "", string)

def remove_http_links(string):
    """
        Removes all http links (ie. http://t.co/0DlGChTBIx) from the string.
    """
    return re.sub(twitter_link_re, "", string)

def remove_pic_links(string):
    """
        Removes all twitter picture links (ie. pic.twitter.com/UTYOLo7wGF) from
        the string.
    """
    return re.sub(twitter_pic_link_re, "", string)

def extract_http_links(string):
    """
        Returns every http link (ie. http://t.co/0DlGChTBIx) in the order in
        which they occur in the string.
    """
    return re.findall(twitter_link_re, string)

def extract_pic_links(string):
    """
        Returns every twitter picture link (ie. pic.twitter.com/UTYOLo7wGF) in
        the order in which they occur in the string.
    """
    return re.findall(twitter_pic_link_re, string)

def extract_hts(string):
    """
        Returns every twitter hashtag (ie. #MakeAmericaGreatAgain) in the order
        in which they occur in the string.
    """
    hts = []
    for ht in re.findall(twitter_ht_re, string):
        hts.append('#' + ht)
    return hts

def remove_hts(string):
    """
        Removes all twitter hashtags (ie. #MakeAmericaGreatAgain) from the
        string.
    """
    return re.sub(twitter_ht_re, "", string)

def find_nearest_string(string, candidates):
    """
        Find string candidate most near string passed.
    """
    if string in candidates:
        return string

    most_similar_candidate = None
    most_similar_candidate_score = -1

    for candidate in candidates:
        this_candidate_score = SequenceMatcher(None, candidate, string).ratio()
        if this_candidate_score > most_similar_candidate_score:
            most_similar_candidate = candidate
            most_similar_candidate_score = this_candidate_score

    return most_similar_candidate

def remove_at_prefixes(string):
    """
        Returns string with all initial twitter handles (ie. @realDonaldTrump)
        removed. Only initial handles will be removed. All extraneous spaces
        will be removed as well.
    """

    if len(string) <= 1:
        return string

    ats = set(extract_ats(string))
    split_string = string.split()

    # If string only contains ats, return null string
    if set(split_string) <= ats:
        return ''

    while split_string[0] in ats:
        split_string = split_string[1:]
    return ' '.join(split_string)

def get_random_string(length):
    """
        Generate random string of desired length.

        Credit:
        https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))
