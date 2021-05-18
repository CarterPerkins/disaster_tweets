import re
import string

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)


def clean_tweet(tweet: str) -> str:
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+', '', tweet)  # remove url
    lemmatizer = WordNetLemmatizer()
    tweet_tokenizer = TweetTokenizer()

    cleaned_tweet = []
    for token in tweet_tokenizer.tokenize(tweet):
        # skip usernames, stopwords, and punctuation
        if token[0] == '@' or token in STOPWORDS \
                or token in PUNCTUATION:
            continue
        # convert hashtags
        if token[0] == '#':
            token = token[1:]

        # lemmatize words
        token = lemmatizer.lemmatize(token)
        cleaned_tweet.append(token)

    return ' '.join(cleaned_tweet)


def preprocess() -> pd.DataFrame:
    df = pd.read_csv('./data/train.csv')
    df['text'] = df['text'].apply(clean_tweet)
    df = df[['text', 'target']]
    return df
