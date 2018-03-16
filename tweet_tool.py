from credentials import *
import tweepy
from datetime import *
import numpy as np
import time

"""
Assumes file credentials.py contains the following definitions:

consumer_key
consumer_secret
access_token
access_token_secret

Credit to https://www.digitalocean.com/community/tutorials/how-to-create-a-twitterbot-with-python-3-and-the-tweepy-library
"""
class TweetTool:

    def __init__(self):

        # Access and authorize our Twitter credentials from credentials.py
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    # Return tweets from user
    def get_tweets(self, user, count=10):
        return self.api.user_timeline(user, count=count)

    def tweet(self, contents):
        self.api.update_status(contents)

    def get_num_followers(self, user):
        return self.api.get_user(user).followers_count

    @staticmethod
    def get_times(n, minute_avg, minute_std, prime_time=True):

        minute = timedelta(minutes=1)
        times = []
        current_time = datetime.today()

        count = 0

        while count < n:

            # Prime time setting will not use times outside of window
            if prime_time:
                if current_time.hour > 8 and current_time.hour < 20:
                    times.append(current_time)
                    count += 1
            else:
                times.append(current_time)
                count += 1

            minutes = abs(int(np.random.normal(minute_avg, minute_std))) * minute
            current_time += minutes
        return times

    def schedule_tweets(self, tweets, times):

        if len(tweets) != len(times):
            raise ValueError

        print(len(tweets), 'tweets')
        print('Last tweet scheduled at', times[-1])
        print()

        for tweet, scheduled_time in zip(tweets, times):
            print(tweet)
            print('Waiting until', scheduled_time, 'to tweet....')

            while datetime.now() < scheduled_time:
                time.sleep(60)

            self.tweet(tweet)
            print('Done.')
            print()
