from parse_tools import ParseTools
import unittest
import string
import re

class TestParseTool(unittest.TestCase):

    def test_fix_ats(self):
        test_at_set = {'@realDonaldTrump', '@joe'}
        self.assertEqual(ParseTools.fix_ats('Tweet with @realDOOOONALDTRUMP', test_at_set), 'Tweet with @realDonaldTrump')
        self.assertEqual(ParseTools.fix_ats('Tweet with @jump', test_at_set), 'Tweet with @joe')
        self.assertEqual(ParseTools.fix_ats('Tweet with no ats', test_at_set), 'Tweet with no ats')
        self.assertEqual(ParseTools.fix_ats('', test_at_set), '')

    def test_contains_letters(self):
        self.assertTrue(ParseTools.contains_letters("String with letters"))
        self.assertFalse(ParseTools.contains_letters(".!.?"))
        self.assertFalse(ParseTools.contains_letters("."))
        self.assertFalse(ParseTools.contains_letters("//"))
        self.assertFalse(ParseTools.contains_letters(""))

    def test_get_count_funcs(self):
        test_words = ['John', 'Jacob']
        test_funcs = ParseTools.get_count_funcs(test_words)
        self.assertEqual(len(re.findall('John', "String with John twice John")), 2)
        self.assertEqual(test_funcs[0].__name__, 'count-John')
        self.assertEqual(test_funcs[1].__name__, 'count-Jacob')
        self.assertEqual(test_funcs[0]("String with John twice John"), 2)
        self.assertEqual(test_funcs[1]("String with John twice John"), 0)

    def test_length_n_sequences(self):
        test_char = '.'
        test_n = 3
        self.assertEqual(ParseTools.length_n_sequences("This is a string a sequence of three periods...", char=test_char, n=test_n), ["..."])
        self.assertEqual(ParseTools.length_n_sequences("This is a string a sequence of two periods..", char=test_char, n=test_n), [])
        self.assertEqual(ParseTools.length_n_sequences("This is a string a sequence of four periods....", char=test_char, n=test_n), ["...."])
        self.assertEqual(ParseTools.length_n_sequences("This is a string... has two period sequences...", char=test_char, n=test_n), ["...", "..."])
        self.assertEqual(ParseTools.length_n_sequences("This is a string no periods", char=test_char, n=test_n), [])
        self.assertEqual(ParseTools.length_n_sequences("............", char=test_char, n=test_n), ["............"])

    def test_replace_ats_strings_with_ats(self):
        self.assertEqual(ParseTools.replace_ats("String with an @ @realDonaldTrump", "Jerry"), "String with an @ Jerry")

    def test_replace_ats_strings_without_ats(self):
        self.assertEqual(ParseTools.replace_ats("String without an @", "Jerry"), "String without an @")

    def test_is_proper_sentence_propers(self):
        """
            Verify that proper sentences (ones that start with a capital letter and end with a punctuation) can be identified as such.
        """
        self.assertTrue(ParseTools.is_proper_sentence("Regular sentence."))
        self.assertTrue(ParseTools.is_proper_sentence("Regular sentence..."))
        self.assertTrue(ParseTools.is_proper_sentence("Regular sentence?"))
        self.assertTrue(ParseTools.is_proper_sentence("Regular sentence!"))

    def test_is_proper_sentence_inpropers(self):
        """
            Verify that inproper sentences (ones that do not start with a capital letter or do not end with a punctuation) can be identified as such.
        """
        self.assertFalse(ParseTools.is_proper_sentence(""))
        self.assertFalse(ParseTools.is_proper_sentence("."))
        self.assertFalse(ParseTools.is_proper_sentence(".."))
        self.assertFalse(ParseTools.is_proper_sentence("..."))
        self.assertFalse(ParseTools.is_proper_sentence("...."))
        self.assertFalse(ParseTools.is_proper_sentence("Sentence without punctuation"))
        self.assertFalse(ParseTools.is_proper_sentence("sentence without an initial capital letter."))
        self.assertFalse(ParseTools.is_proper_sentence("sentence without an initial capital letter and no punctuation"))

    def test_replace_all_replaced(self):
        """
            Verify that replace_all can correctly replace characters in the string
            that are not in meant to be kept.
        """
        self.assertEqual(ParseTools.replace_all(string.printable, " ", "⚾️"), "  ")
        self.assertEqual(ParseTools.replace_all(string.printable, " ", "🇺🇸"), "  ")

    def test_replace_all_not_replaced(self):
        """
            Verify that replace_all can correctly leave string with all characters present in the keeps unaltered.
        """
        self.assertEqual(ParseTools.replace_all(string.printable, " ", string.printable), string.printable)
        self.assertEqual(ParseTools.replace_all(string.printable, " ", "a"), "a")

    def test_split_join_no_changes(self):
        """
            Verify that using split/join string combo does
            not alter the string if ones space was between each word and nowhere
            else.
        """
        self.assertEqual(ParseTools.split_join("string with words"), "string with words")
        self.assertEqual(ParseTools.split_join("word"), "word")
        self.assertEqual(ParseTools.split_join("This is a complete sentence."), "This is a complete sentence.")
        self.assertEqual(ParseTools.split_join("This is a complete sentence (and some extra characters)."), "This is a complete sentence (and some extra characters).")
        self.assertEqual(ParseTools.split_join("Do you want to go to the mall? Sure!"), "Do you want to go to the mall? Sure!")

    def test_split_join_extra_space_remove(self):
        """
            Verify that using split/join string combo removes
            all spaces that are not in between words.
        """
        self.assertEqual(ParseTools.split_join("string with extra space at end "), "string with extra space at end")
        self.assertEqual(ParseTools.split_join(" word "), "word")
        self.assertEqual(ParseTools.split_join("string  with  extra     spaces"), "string with extra spaces")

    def test_split_join_alt_chars(self):
        """
            Verify that using split/join string combo removes
            all space-like extra characters (carriage return, tab, etc.)
        """
        self.assertEqual(ParseTools.split_join("string with carriage \n return"), "string with carriage return")
        self.assertEqual(ParseTools.split_join("string with \t tab"), "string with tab")
        self.assertEqual(ParseTools.split_join("string with carriage return\n"), "string with carriage return")
        self.assertEqual(ParseTools.split_join("string with tab\t"), "string with tab")

    def test_punctuate_space_last_char_punctuation_applied(self):
        """
            Verify that punctuate adds punctuation when last character in string is space, but previous character is not punctuation.
        """
        original_string = "String with character-space end "
        expected_outputs = {original_string[:-1] + '.', original_string[:-1] + '?', original_string[:-1] + '!'}
        self.assertIn(ParseTools.punctuate(original_string), expected_outputs)

    def test_punctuate_space_last_char_punctuation_not_applied(self):
        """
            Verify that punctuate does not add punctuation when last character in string is space, but previous character is punctuation (last space should still be removed).
        """
        self.assertEqual(ParseTools.punctuate("String with already present last punctuation then space! "), "String with already present last punctuation then space!")
        self.assertEqual(ParseTools.punctuate("String with already present last punctuation then space. "), "String with already present last punctuation then space.")
        self.assertEqual(ParseTools.punctuate("String with already present last punctuation then space? "), "String with already present last punctuation then space?")

    def test_punctuate_non_space_last_char_punctuation_applied(self):
        """
            Verify that punctuate removes last word and applies punctuation when the last character in the string is neither a space nor a punctuation.
        """
        original_string = "String with character end z"
        expected_outputs = {"String with character end" + '.', "String with character end" + '?', "String with character end" + '!'}
        self.assertIn(ParseTools.punctuate(original_string), expected_outputs)

    def test_punctuate_non_space_last_char_punctuation_not_applied(self):
        """
            Verify that punctuate leaves string alone when last char is punctuation.
        """
        self.assertEqual(ParseTools.punctuate("String with already present last punctuation!"), "String with already present last punctuation!")
        self.assertEqual(ParseTools.punctuate("String with already present last punctuation."), "String with already present last punctuation.")
        self.assertEqual(ParseTools.punctuate("String with already present last punctuation?"), "String with already present last punctuation?")
        self.assertEqual(ParseTools.punctuate("."), ".")
        self.assertEqual(ParseTools.punctuate("!"), "!")
        self.assertEqual(ParseTools.punctuate("?"), "?")

    def test_punctuate_edge_cases(self):
        """
            Veify that edge cases are handled correctly by punctuate (single char strings, strings with just space-like characters, etc.)
        """
        self.assertIn(ParseTools.punctuate(''), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('a'), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate(')'), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('\n'), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('\t'), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate(' '), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('\n   '), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('\t   '), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('    '), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('    \n'), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('    \t'), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('    \n    '), {'.', '?', '!'})
        self.assertIn(ParseTools.punctuate('    \t   '), {'.', '?', '!'})

    def test_remove_dots(self):
        self.assertEqual(ParseTools.remove_dots("String with three dot pattern…"), "String with three dot pattern")
        self.assertEqual(ParseTools.remove_dots("String with three dot pattern …"), "String with three dot pattern ")

    def test_extract_apostrophe_words_string_with_apostrophe_words(self):
        self.assertEqual(ParseTools.extract_apostrophe_words("This sentence's got a word with an apostrophe"), ["sentence's"])
        self.assertEqual(ParseTools.extract_apostrophe_words("doesn't"), ["doesn't"])
        self.assertEqual(ParseTools.extract_apostrophe_words("'don't'"), ["don't"])

    def test_extract_apostrophe_words_string_without_apostrophe_words(self):
        self.assertEqual(ParseTools.extract_apostrophe_words(""), [])
        self.assertEqual(ParseTools.extract_apostrophe_words("This sentence has no words with an apostrophe"), [])
        self.assertEqual(ParseTools.extract_apostrophe_words("and'"), [])
        self.assertEqual(ParseTools.extract_apostrophe_words("'and"), [])

    def test_is_quoted_tweet_on_quoted_tweets(self):

        self.assertTrue(ParseTools.is_quoted_tweet("\"@VeryOddDog: What\'s BRUTAL is a nation WITHOUT Trump!\""))
        self.assertTrue(ParseTools.is_quoted_tweet("\"@realDonaldTrump: Entrepreneurs: Being stubborn is a big part of being a winner. Never give up!\""))
        self.assertTrue(ParseTools.is_quoted_tweet("\"@thequote: Without passion you don't have energy, without energy you have nothing. - Donald Trump\""))
        self.assertTrue(ParseTools.is_quoted_tweet("\"@thequote: Without passion you don't have energy, without energy you have nothing. - Donald Trump\" Thanks!"))
        self.assertTrue(ParseTools.is_quoted_tweet("\'@thequote: Without passion you don't have energy, without energy you have nothing. - Donald Trump\'"))
        self.assertTrue(ParseTools.is_quoted_tweet("\'@thequote: Without passion you don't have energy, without energy you have nothing. - Donald Trump\' Thanks!"))

    def test_is_quoted_tweet_on_regular_strings(self):

        self.assertFalse(ParseTools.is_quoted_tweet("\"String surrounded by double quotes\""))
        self.assertFalse(ParseTools.is_quoted_tweet("\'String surrounded single quotes\'"))
        self.assertFalse(ParseTools.is_quoted_tweet("@realDonaldTrump: Missing surrounding quotes"))
        self.assertFalse(ParseTools.is_quoted_tweet("@thequote: Missing surrounding quotes"))

    def test_is_quoted_tweet_on_invalid_strings(self):
        self.assertFalse(ParseTools.is_quoted_tweet(""))
        self.assertFalse(ParseTools.is_quoted_tweet("\""))
        self.assertFalse(ParseTools.is_quoted_tweet("\'"))
        self.assertFalse(ParseTools.is_quoted_tweet("@"))
        self.assertFalse(ParseTools.is_quoted_tweet(":"))

    def test_extract_sentences_regular(self):
        """
            Verify that extract_sentences can be correctly applied to strings with sentences whose only punctuation is at the end.
        """

        # Strings with one grammatically CORRECT sentence
        self.assertEqual(ParseTools.extract_sentences("Here is a declaritive sentence."), ["Here is a declaritive sentence."])
        self.assertEqual(ParseTools.extract_sentences("Is this an interrogative sentence?"), ["Is this an interrogative sentence?"])
        self.assertEqual(ParseTools.extract_sentences("This is an excalamtory sentence!"), ["This is an excalamtory sentence!"])

        # Strings with multiple grammatically CORRECT sentences
        self.assertEqual(ParseTools.extract_sentences("This string has two sentences. Both of the sentences are grammatically correct."), ["This string has two sentences.", "Both of the sentences are grammatically correct."])

        # Strings with multiple grammatically INCORRECT sentences
        self.assertEqual(ParseTools.extract_sentences("Sentences have this does two. Grammar not is."), ["Sentences have this does two.", "Grammar not is."])

    def test_extract_sentences_twitter_words(self):
        """
            Verify behavior of extract_sentences when it enconters twitter words.
        """
        self.assertEqual(ParseTools.extract_sentences("@TrumpPeeLannin I am there and and you can see the proof?"), ["@TrumpPeeLannin I am there and and you can see the proof?"])
        self.assertEqual(ParseTools.extract_sentences("@TheBearthen   Thanks!"), ["@TheBearthen   Thanks!"])

    def test_extract_sentences_alt_punctuation(self):
        """
            Verify behavior of extract_sentences on strings with alternative punctuation.
        """
        self.assertEqual(ParseTools.extract_sentences("."), ["."])
        self.assertEqual(ParseTools.extract_sentences(".."), [".."])
        self.assertEqual(ParseTools.extract_sentences("..."), ["..."])
        self.assertEqual(ParseTools.extract_sentences("...."), ["...."])

        self.assertEqual(ParseTools.extract_sentences("This string has no punctuation"), ["This string has no punctuation"])
        self.assertEqual(ParseTools.extract_sentences("This sentence has punctuation. This does not"), ["This sentence has punctuation.", "This does not"])
        self.assertEqual(ParseTools.extract_sentences("This sentence ends in three dots... This does not."), ["This sentence ends in three dots...", "This does not."])

    def test_extract_sentences_alternative_puncuation(self):
        """
            Verify that extract_sentences can be correctly applied to strings with sentences with embedded punctuation.
        """
        self.assertEqual(ParseTools.extract_sentences("Dr. Seuss was a good author."), ["Dr. Seuss was a good author."])

    def test_extract_sentences_null_string(self):
        """
            Verify that extract_sentences can correctly indicate that a null string has no sentences.
        """
        self.assertEqual(ParseTools.extract_sentences(""), [])

    def test_extract_words_regular_words(self):
        """
            Verify that extract_words can correctly pick out words in a string following common english rules.
        """
        self.assertEqual(ParseTools.extract_words("This string has words."), ["This", "string", "has", "words"])
        self.assertEqual(ParseTools.extract_words("Nonsense word: asd"), ["Nonsense", "word", "asd"])
        self.assertEqual(ParseTools.extract_words("Dashed-phrase"), ["Dashed", "phrase"])

    def test_extract_words_apostrophes(self):
        self.assertEqual(ParseTools.extract_words("doesn't"), ["doesn't"])
        self.assertEqual(ParseTools.extract_words("Trump's"), ["Trump's"])
        self.assertEqual(ParseTools.extract_words("it's"), ["it's"])
        self.assertEqual(ParseTools.extract_words("its'"), ["its"])

    def test_extract_words_quoted_words(self):
        self.assertEqual(ParseTools.extract_words("String with a 'quoted' word"), ["String", "with", "a", "quoted", "word"])
        self.assertEqual(ParseTools.extract_words("String with a 'quoted' word and don't"), ["String", "with", "a", "quoted", "word", "and", "don't"])
        self.assertEqual(ParseTools.extract_words("String with 'quoted' words 'don't'"), ["String", "with", "quoted", "words", "don't"])

    def test_extract_words_quoted_phrases(self):
        self.assertEqual(ParseTools.extract_words("String with a 'quoted phrase'"), ["String", "with", "a", "quoted", "phrase"])

    def test_extract_words_twitter_words(self):
        """
            Verify that extract_words can correctly pick out twitter words from a string (ie. usernames, links, hashtags).
        """
        self.assertEqual(ParseTools.extract_words("This string contains a twitter handle: @realDonaldTrump"), ["This", "string", "contains", "a", "twitter", "handle", "@realDonaldTrump"])
        self.assertEqual(ParseTools.extract_words("This string contains a twitter link: http://t.co/0DlGChTBIx"), ["This", "string", "contains", "a", "twitter", "link", "http://t.co/0DlGChTBIx"])
        self.assertEqual(ParseTools.extract_words("This string contains a twitter link and handle: http://t.co/0DlGChTBIx @realDonaldTrump"), ["This", "string", "contains", "a", "twitter", "link", "and", "handle", "http://t.co/0DlGChTBIx", "@realDonaldTrump"])
        self.assertEqual(ParseTools.extract_words("This string contains a twitter hashtag: #MakeAmericaGreatAgain"), ["This", "string", "contains", "a", "twitter", "hashtag", "#MakeAmericaGreatAgain"])
        self.assertEqual(ParseTools.extract_words("This string contains a twitter username followed by a colon @realDonaldTrump:"), ["This", "string", "contains", "a", "twitter", "username", "followed", "by", "a", "colon", "@realDonaldTrump"])
        self.assertEqual(ParseTools.extract_words("This string contains a twitter username with a trailing period .@realDonaldTrump"), ["This", "string", "contains", "a", "twitter", "username", "with", "a", "trailing", "period", "@realDonaldTrump"])

    def test_extract_words_null_string(self):
        """
            Verify that extract_words correctly extracts nothing from the null string.
        """
        self.assertEqual(ParseTools.extract_words(""), [])

    def test_extract_hts_strings_with_hts(self):
        """
            Verify that extract_hts correctly returns the hashtags in the string in the order in which they occur.
        """
        self.assertEqual(ParseTools.extract_hts("String with a few #Hash #Tags"), ["#Hash", "#Tags"])
        self.assertEqual(ParseTools.extract_hts("String with a #Hashtags in weird #Places"), ["#Hashtags", "#Places"])

    def test_extract_hts_strings_without_hts(self):
        """
            Verify that extract_hts correctly indicates that no hashtags exist in a string.
        """
        self.assertEqual(ParseTools.extract_hts("String with no hastags"), [])

    def test_remove_hts_strings_with_hts(self):
        """
            Verify that remove_hts correctly removes hashtags from string.
        """
        self.assertEqual(ParseTools.remove_hts("String with a few #Hash #Tags"), "String with a few  ")

    def test_remove_hts_strings_without_hts(self):
        """
            Verify that remove_hts does not alter string without hashtags.
        """
        self.assertEqual(ParseTools.remove_hts("String without any hashtags"), "String without any hashtags")

    def test_extract_ats_strings_with_ats(self):
        """
            Verify that extract_ats returns twitter usernames from string in the order in which they occur.
        """
        self.assertEqual(ParseTools.extract_ats("String with a few @twitter @names"), ["@twitter", "@names"])
        self.assertEqual(ParseTools.extract_ats("@twitter String with names @cool in wide varitey of places @names"), ["@twitter", "@cool", "@names"])

    def test_extract_ats_strings_without_ats(self):
        """
            Verify that extract_ats correctly indicates that a string has no twitter usernames.
        """
        self.assertEqual(ParseTools.extract_ats("String with no twitter names"), [])
        self.assertEqual(ParseTools.extract_ats("String with a twitter name without the at: realDonaldTrump"), [])
        self.assertEqual(ParseTools.extract_ats("String with just the @ symbol"), [])
        self.assertEqual(ParseTools.extract_ats("String with name like phrase @000"), [])

    def test_extract_ats_at_with_symbols(self):
        """
            Verify that extract_ats can catch twitter names when they are surrounded by symbols.
        """
        self.assertEqual(ParseTools.extract_ats("String @realDonaldTrump: that has twitter name-colon combo"), ["@realDonaldTrump"])
        self.assertEqual(ParseTools.extract_ats(".@realDonaldTrump String that has twitter name-period combo"), ["@realDonaldTrump"])

    def test_remove_ats_strings_with_ats(self):
        """
            Verify that remove_ats removes twitter usernames from a string.
        """
        self.assertEqual(ParseTools.remove_ats("String with a few @twitter @names"), "String with a few  ")

    def test_remove_ats_strings_without_ats(self):
        """
            Verify that remove_ats leaves a string without twitter usernames unchanged.
        """
        self.assertEqual(ParseTools.remove_ats("String with no twitter names"), "String with no twitter names")

    def test_remove_pic_links_strings_without_pic_links(self):
        """
            Verify that remove_pic_links leaves a string without twitter picture links unchanged.
        """
        self.assertEqual(ParseTools.remove_pic_links("String with no twitter picture links"), "String with no twitter picture links")

    def test_remove_pic_links_strings_with_pic_links(self):
        """
            Verify that remove_pic_links removes all twitter picture links from a string.
        """
        self.assertEqual(ParseTools.remove_pic_links("String with a twitter picture link pic.twitter.com/UTYOLo7wGF"), "String with a twitter picture link ")

    def test_remove_http_links_strings_without_http_links(self):
        """
            Verify that remove_http_links leaves a string without twitter http links unchanged.
        """
        self.assertEqual(ParseTools.remove_http_links("String with no http links"), "String with no http links")

    def test_remove_http_links_strings_with_http_links(self):
        """
            Verify that remove_http_links removes all twitter http links from a string.
        """
        self.assertEqual(ParseTools.remove_http_links("String with an http link http://t.co/0DlGChTBIx"), "String with an http link ")

    def test_remove_pic_links_embedded(self):
        """
            Verify that remove_pic_links can identify and remove a twitter picture link embedded in other text.
        """
        self.assertEqual(ParseTools.remove_pic_links("String with an embedded twitter picture link:pic.twitter.com/UTYOLo7wGF"), "String with an embedded twitter picture link:")

    def test_remove_http_links_embedded(self):
        """
            Verify that remove_http_links can identify and remove a twitter http link embedded in other text.
        """
        self.assertEqual(ParseTools.remove_http_links("String with an embedded http link:http://t.co/0DlGChTBIx"), "String with an embedded http link:")

    def test_find_nearest_string_string_in_candidates(self):
        """
            Verify that find_nearest_string can correctly match a string in the candidates with itself.
        """
        test_candidates = {"Oranges", "Tomatoes", "Grapes"}

        self.assertEqual(ParseTools.find_nearest_string("Oranges", test_candidates), "Oranges")
        self.assertEqual(ParseTools.find_nearest_string("Tomatoes", test_candidates), "Tomatoes")
        self.assertEqual(ParseTools.find_nearest_string("Grapes", test_candidates), "Grapes")

    def test_find_nearest_string_string_not_in_candidates(self):
        """
            Verify that find_nearest_string can correctly match a string with its closest candidate.
        """
        test_candidates = {"Oranges", "Tomatoes", "Grapes"}

        self.assertEqual(ParseTools.find_nearest_string("oranges", test_candidates), "Oranges")
        self.assertEqual(ParseTools.find_nearest_string("iranges", test_candidates), "Oranges")
        self.assertEqual(ParseTools.find_nearest_string("Oranges ", test_candidates), "Oranges")
        self.assertEqual(ParseTools.find_nearest_string(" Oranges", test_candidates), "Oranges")
        self.assertEqual(ParseTools.find_nearest_string("Orion", test_candidates), "Oranges")

        self.assertEqual(ParseTools.find_nearest_string("Tommmmatoes", test_candidates), "Tomatoes")
        self.assertEqual(ParseTools.find_nearest_string("Bomatoes   ", test_candidates), "Tomatoes")

    def test_find_nearest_string_on_ats(self):
        """
            Verify that closest twitter username candidate can be found.
        """
        test_candidates = {"@realDonalTrump", "@FES", "@jack"}

        self.assertEqual(ParseTools.find_nearest_string("@trump", test_candidates), "@realDonalTrump")
        self.assertEqual(ParseTools.find_nearest_string("@j", test_candidates), "@jack")

    def test_get_random_string_lengths(self):
        """
            Verify that the random strings produced by get_random_string contain the expected number of characters.
        """
        self.assertEqual(len(ParseTools.get_random_string(0)), 0)
        self.assertEqual(len(ParseTools.get_random_string(1)), 1)
        self.assertEqual(len(ParseTools.get_random_string(100)), 100)

    def test_remove_at_prefixes_prefixed_strings(self):
        """
            Verify that remove_at_prefixes can remove initial twitter usernames from a string.
        """
        self.assertEqual(ParseTools.remove_at_prefixes("@realDonalTrump a tweet with an initial at prefix"), "a tweet with an initial at prefix")
        self.assertEqual(ParseTools.remove_at_prefixes("@realDonalTrump @FES a tweet with initial at prefixes"), "a tweet with initial at prefixes")
        self.assertEqual(ParseTools.remove_at_prefixes("@realDonalTrump @FES"), "")
        self.assertEqual(ParseTools.remove_at_prefixes(" @realDonalTrump @FES"), "")

    def test_remove_at_prefixes_non_prefixed_strings(self):
        """
            Verify that remove_at_prefixes will not alter strings that do not begin with twitter usernames.
        """
        self.assertEqual(ParseTools.remove_at_prefixes("Tweet without an at prefix"), "Tweet without an at prefix")
        self.assertEqual(ParseTools.remove_at_prefixes("Tweet with an at @realDonaldTrump but no at prefix"), "Tweet with an at @realDonaldTrump but no at prefix")
        self.assertEqual(ParseTools.remove_at_prefixes("a @realDonalTrump @FES"), "a @realDonalTrump @FES")
        self.assertEqual(ParseTools.remove_at_prefixes("@ a"), "@ a")

    def test_remove_at_prefixes_edge_cases(self):
        """
            Verify that remove_at_prefixes correctly handles edge string cases (null string, single char string, etc.)
        """
        self.assertEqual(ParseTools.remove_at_prefixes(""), "")
        self.assertEqual(ParseTools.remove_at_prefixes("a"), "a")
        self.assertEqual(ParseTools.remove_at_prefixes(" "), " ")
        self.assertEqual(ParseTools.remove_at_prefixes("@"), "@")
        self.assertEqual(ParseTools.remove_at_prefixes("a  "), "a")
        self.assertEqual(ParseTools.remove_at_prefixes("   "), "")
        self.assertEqual(ParseTools.remove_at_prefixes("@  "), "@")

    def test_remove_outer_quotes_quoted_string(self):
        """
            Verify that remove_outer_quotes performs correcly on quoted strings.
        """
        self.assertEqual(ParseTools.remove_outer_quotes('"String with double outer quotes"'), 'String with double outer quotes')
        self.assertEqual(ParseTools.remove_outer_quotes("'String with single outer quotes'"), "String with single outer quotes")

    def test_remove_outer_quotes_invalid_cases(self):
        """
            Verify that remove_outer_quotes leaves alone strings that are not surrounded by two double quotes or single quotes alone.
        """
        self.assertEqual(ParseTools.remove_outer_quotes('String without outer quotes'), 'String without outer quotes')
        self.assertEqual(ParseTools.remove_outer_quotes('String with one outer quote"'), 'String with one outer quote"')
        self.assertEqual(ParseTools.remove_outer_quotes('"String with one outer quote'), '"String with one outer quote')

if __name__ == "__main__":
    unittest.main()
