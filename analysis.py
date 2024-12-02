
import re
import numpy as np
import nltk
import string
from os import getcwd
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def build_freqs(tweets, ys):

    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

import pandas as pd
from nltk.corpus import twitter_samples

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

#train-test split
train_pos = all_positive_tweets[4000:] # 4000 to till the end of the list
test_pos = all_positive_tweets[:4000]# 0 till 3999- these are indexes btw
train_neg = all_negative_tweets[4000:] #same as above
test_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x =test_pos + test_neg
print(len(train_x))

train_y = np.append(np.ones(len(train_pos)),np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)),np.zeros(len(test_neg)))

#building frequency distribution table
freqs= build_freqs(train_x,train_y)

#the sigmoid function for logistic regression
def sigmoid(z):
  h = 1/(1+np.exp(-z))
  return h


def gradientDescent(x, y, theta, alpha, num_iters):

    m = x.shape[0]

    for i in range(0, num_iters):

        # get z, the dot product of x and theta
        z = np.dot(x,theta)


        # get the sigmoid of z
        h =  1/(1+np.exp(-z))

        y = y.reshape(-1, 1)  # Ensure y is (m, 1)
        h = h.reshape(-1, 1)  # Ensure h is (m, 1)

        # calculate the cost function
        J = -1/m * (np.dot(y.transpose(),np.log(h)) + np.dot((1-y).transpose(),np.log(1-h)))
        J = float(J)


        # update the weights theta
        theta = theta - (alpha/m) * np.dot(x.transpose(),(h-y))

    return J, theta

#extracting the features from the tweets
def extract_features(tweet, freqs):

    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    #bias term is set to 1
    x[0,0] = 1

    for word in word_l:

        x[0,1]  += freqs.get((word,1.0),0) #increment the positive word count


        x[0,2] += freqs.get((word,0.0),0) #increment the negative wrod count


    assert(x.shape == (1,3))
    return x


X = np.zeros((len(train_x),3))
for i in range(len(train_x)):
  X[i, :] = extract_features(train_x[i],freqs)

Y = train_y

J, theta = gradientDescent(X,Y, np.zeros((3,1)),1e-9,1500)
print("the cost after training is: ",J )
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

def predict_tweet(tweet, freqs, theta):

    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))



    return y_pred

my_tweet = ("I like this")
print(predict_tweet(my_tweet, freqs, theta))

def test_logistic_regression(test_x, test_y, freqs, theta):

    # the list for storing predictions
    q = 0
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1)

        else:
            # append 0 to the list
            y_hat.append(0)



    accuracy = (y_hat == np.squeeze(test_y)).sum()/len(test_x)


    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy: ", tmp_accuracy)

my_tweet =" Put your tweet here"
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')
