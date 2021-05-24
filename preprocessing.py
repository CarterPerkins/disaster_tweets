import re
import string
from typing import Tuple

import numpy
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split

STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)


def clean_tweet(tweet: str) -> str:
    tweet = tweet.lower()
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')  # remove unicode characters
    tweet = re.sub(r'http\S+', '', tweet)  # remove url
    tweet = re.sub(r'<.*?>', '', tweet)  # remove html
    tweet_tokenizer = TweetTokenizer()
    lemmatizer = WordNetLemmatizer()

    cleaned_tweet = []
    for token in tweet_tokenizer.tokenize(tweet):
        # skip usernames, stopwords, punctuation, and <= 2 character tokens
        if token[0] == '@' or token in STOPWORDS or token in PUNCTUATION or len(token) <= 1:
            continue
        # convert hashtags
        if token[0] == '#':
            token = token[1:]
        # skip tokens with <= 2 letters
        if sum(char.isalpha() for char in token) <= 2:
            continue
        # lemmatize words
        token = lemmatizer.lemmatize(token)
        cleaned_tweet.append(token)

    return ' '.join(cleaned_tweet)

def preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('./data/train.csv')
    df['text'] = df['text'].apply(clean_tweet)
    df = df[['text', 'target']]
    df = df[df.text.str.len() > 0]

    # train test split: 70/15/15
    train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=0)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=0)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return (train_df, val_df, test_df)
