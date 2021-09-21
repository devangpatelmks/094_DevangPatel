# CELL 0

# Import necessary libraries
import nltk
import pandas as pd
import tensorflow as tf
import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples

# CELL 1

nltk.download('twitter_samples')
nltk.download('stopwords')

# CELL 2

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # Remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # Remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # Remove hashtags
    # Only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        # 1 remove stopwords
        if word in stopwords_english:
            continue
        # 2 remove punctuation
        if word in string.punctuation:
            continue
        # 3 stemming word
        word = stemmer.stem(word)
        # 4 Add it to tweets_clean
        tweets_clean.append(word)

    return tweets_clean

# CELL 3

def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)

            #Update the count of pair if present, set it to 1 otherwise
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

# CELL 4

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# CELL 5

# Split the data into two pieces, one for training and one for testing
from sklearn.model_selection import train_test_split

train_pos, test_pos = train_test_split(all_positive_tweets, test_size = 0.3)
train_neg, test_neg = train_test_split(all_negative_tweets, test_size = 0.3)

# Combine positive and negative labels
train_x = train_pos + train_neg
test_x = test_pos + test_neg

# CELL 6

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis = 0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis = 0)

# CELL 7

# Create frequency distribution
freqs = build_freqs(train_x, train_y)
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

# CELL 8

print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))

# CELL 9

def extract_features(tweet, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # tokenizes, stems, and removes stopwords
    output = []
    for word_l in tweet:
        word_l = process_tweet(word_l)

        # 3 elements in the form of a 1 x 3 vector
        x = np.zeros((1, 3))

        #bias term is set to 1
        x[0,0] = 1

        # loop through each word in the list of words
        for word in word_l:

            # increment the word count for the positive label 1
            x[0,1] += freqs.get((word, 1.0),0)

            # increment the word count for the negative label 0
            x[0,2] += freqs.get((word, 0.0),0)


        assert(x.shape == (1, 3))
        output.append(x)
    return output

# CELL 10

final_model = tf.keras.models.Sequential([ tf.keras.layers.Dense(2, activation=tf.nn.softmax) ])
final_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.fit(tf.convert_to_tensor(extract_features(train_x, freqs)), train_y, epochs=1000)

# CELL 11

final_model.evaluate(tf.convert_to_tensor(extract_features(test_x, freqs)), test_y)