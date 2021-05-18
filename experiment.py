from typing import Tuple

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def run_experiment(df: pd.DataFrame, use_val: bool = False) -> np.ndarray:
    def split_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df['text'].to_numpy()
        y = df['target'].to_numpy()
        return (X, y)

    if use_val:
        train_df, val_test_df = train_test_split(df, test_size=0.3,
                                                 random_state=0)
        val_df, test_df = train_test_split(val_test_df, test_size=0.666,
                                           random_state=0)
        X_train, y_train = split_dataframe(train_df)
        X_val, y_val = split_dataframe(val_df)
        X_test, y_test = split_dataframe(test_df)
    else:
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
        X_train, y_train = split_dataframe(train_df)
        X_test, y_test = split_dataframe(test_df)

        cv = CountVectorizer()
        X_train_counts = cv.fit_transform(X_train).toarray()
        X_test_counts = cv.transform(X_test).toarray()

        gnb = GaussianNB()
        model = gnb.fit(X_train_counts, y_train)
        y_pred = model.predict(X_test_counts)

        target_names = ['fake', 'real']
        print(classification_report(y_test, y_pred, target_names=target_names))

        tt = TfidfTransformer()
        X_train_tfidf = tt.fit_transform(X_train_counts).toarray()
        X_test_tfidf = tt.transform(X_test_counts).toarray()

        gnb = GaussianNB()
        model = gnb.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        print(classification_report(y_test, y_pred, target_names=target_names))
