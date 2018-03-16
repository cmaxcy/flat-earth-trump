import unittest
from tweet_tool import TweetTool

class TweetToolTest(unittest.TestCase):

    # Verify that TweetTool object can be created without error
    def test_constructor(self):
        TweetTool()

    # Verify that most recent tweets from a user can be returned
    def test_get_tweets(self):
        test_tweet_tool = TweetTool()
        test_tweets = test_tweet_tool.get_tweets("realDonaldTrump", 100)
        self.assertEqual(len(test_tweets), 100)

if __name__ == "__main__":
    unittest.main()
