from typing import Dict

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from utility import split_dataframe


def bag_of_words(df: pd.DataFrame) -> Dict:
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
    X_train, y_train = split_dataframe(train_df)
    X_test, y_test = split_dataframe(test_df)

    # Naive Bayes with Bag of Words
    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train).toarray()
    X_test_counts = cv.transform(X_test).toarray()

    gnb = GaussianNB()
    model = gnb.fit(X_train_counts, y_train)
    y_pred = model.predict(X_test_counts)

    return {'roc_auc_score': roc_auc_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)}

def tf_idf(df: pd.DataFrame) -> Dict:
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
    X_train, y_train = split_dataframe(train_df)
    X_test, y_test = split_dataframe(test_df)

    # Naive Bayes with Bag of Words
    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train).toarray()
    X_test_counts = cv.transform(X_test).toarray()

    # Naive Bayes with TF-IDF Vectors
    tt = TfidfTransformer()
    X_train_tfidf = tt.fit_transform(X_train_counts).toarray()
    X_test_tfidf = tt.transform(X_test_counts).toarray()

    gnb = GaussianNB()
    model = gnb.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    return {'roc_auc_score': roc_auc_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)}

def run_naive_bayes(df: pd.DataFrame, mode: str) -> Dict:
    callbacks = {
        'bag_of_words': lambda: bag_of_words(df),
        'tf_idf': lambda: tf_idf(df)}

    return callbacks[mode]()

