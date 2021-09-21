# CELL 0

# Import necessary libraries
import re
import string
import numpy as np
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples 

# CELL 1

# Download dataset
nltk.download('twitter_samples')
nltk.download('stopwords')

# CELL 2

"""process_tweet(): cleans the text, tokenizes it into separate words, removes
   stopwords, and converts words to stems."""
def process_tweet(tweet):
    
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet"""
    
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
    
    # Tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
            # 1 Remove stopwords
            # 2 Remove punctuation
            if (word not in stopwords_english and word not in string.punctuation):
                # 3 Stemming word
                stem_word = stemmer.stem(word)
                # 4 Add it to tweets_clean
                tweets_clean.append(stem_word)

    return tweets_clean

# CELL 3

"""build_freqs counts how often a word in the 'corpus' (the entire set of tweets) was associated with
   a positive label '1' or a negative label '0', then builds the freqs dictionary, where each key is
   a (word,label) tuple, and the value is the count of its frequency within the corpus of tweets."""

def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency"""
    
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
            
            # Update the count of pair if present, set it to 1 otherwise
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
            
    return freqs

# CEll 4

# Prepare the data
# The twitter_samples contains subsets of 5,000 positive tweets, 5,000 negative tweets, and
# the full set of 10,000 tweets.

# Select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

print(len(all_positive_tweets))
print(len(all_negative_tweets))

# CELL 5

# Split the data into two pieces, one for training and one for testing

train_len = int((len(all_positive_tweets) * 0.8))

test_pos = all_positive_tweets[train_len:]
train_pos = all_positive_tweets[:train_len]
test_neg = all_negative_tweets[train_len:]
train_neg = all_negative_tweets[:train_len]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

# CELL 6

# Combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# CELL 7

# Create frequency dictionary
freqs = build_freqs(train_x, train_y)

# Check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

"""HERE, The freqs dictionary is the frequency dictionary that's being built.
   The key is the tuple (word, label), such as ("happy",1) or ("happy",0).
   The value stored for each key is the count of how many times the word "happy"
   was associated with a positive label, or how many times "happy" was associated
   with a negative label."""

# CELL 8

# Process tweet
# Example
print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))

# CELL 9

# Logistic regression :
# Sigmoid


def sigmoid(z): 
       
    # calculate the sigmoid of z
    h = 1/(1+np.exp(-z))
    
    return h

# CELL 10

def gradientDescent(x, y, theta, alpha, num_iters):
  
    # get 'm', the number of rows in matrix x
    m = len(x)
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = (-1/m)*(y.T @ np.log(h) + (1-y).T @ np.log(1-h))

        # update the weights theta
        theta = theta - (alpha/m) * np.dot(x.transpose(),(h-y))
        
    J = float(J)
    return J, theta

# CELL 11

"""Extracting the features
   Given a list of tweets, extract the features and store them in a matrix. You will extract two features.
   The first feature is the number of positive words in a tweet.
   The second feature is the number of negative words in a tweet.
   Then train your logistic regression classifier on these features.
   Test the classifier on a validation set."""

def extract_features(tweet, freqs):
    
    """
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    """

    # tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    # bias term is set to 1
    x[0,0] = 1 
        
    # Loop through each word in the list of words
    for word in word_l:
        
        # Increment the word count for the positive label 1
        x[0,1] += freqs.get((word, 1.0),0)
        
        # Increment the word count for the negative label 0
        x[0,2] += freqs.get((word, 0.0),0)
        
    
    assert(x.shape == (1, 3))
    return x

# CELL 12

# Check the function

# Test 1
# Test on training data
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)

# CELL 13

# Test 2:
# Check for when the words are not in the freqs dictionary
tmp2 = extract_features('Devang Bhalala', freqs)
print(tmp2)

# CELL 14

# Training Your Model
# To train the model:

# Stack the features for all training examples into a matrix X. Call gradientDescent

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")

# CELL 15

def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    
    # extract the features of the tweet and store it into x
    x = extract_features(tweet,freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))
    
    
    return y_pred

# CELL 16

# Run this cell to test your function
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))

# CELL 17

# Check performance using the test set
def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    
    # The list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # Get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # Append 1.0 to the list
            y_hat.append(1)
        else:
            # Append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # Convert both to one-dimensional arrays in order to compare them using the '==' operator
    count=0
    y_hat=np.array(y_hat)
    m=len(test_y)
    #print(m)
    
    test_y=np.reshape(test_y,m)
    #print(y_hat.shape)
    #print(test_y.shape)
    
    accuracy = ((test_y == y_hat).sum())/m
    
    return accuracy

# CELL 18

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")