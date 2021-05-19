from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB

from utility import split_dataframe


def bag_of_words(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Naive Bayes with Bag of Words
    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train).toarray()
    X_test_counts = cv.transform(X_test).toarray()

    return (X_train_counts, X_test_counts)

def tf_idf(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train_counts, X_test_counts = bag_of_words(X_train, X_test)

    # Naive Bayes with TF-IDF Vectors
    tt = TfidfTransformer()
    X_train_tfidf = tt.fit_transform(X_train_counts).toarray()
    X_test_tfidf = tt.transform(X_test_counts).toarray()

    return (X_train_tfidf, X_test_tfidf)

def run_naive_bayes(df: pd.DataFrame, mode: str, param_grid: Dict) -> Dict:
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
    X_train, y_train = split_dataframe(train_df)
    X_test, y_test = split_dataframe(test_df)

    callbacks = {
        'bag_of_words': lambda: bag_of_words(X_train, X_test),
        'tf_idf': lambda: tf_idf(X_train, X_test)}

    X_train, X_test = callbacks[mode]()
    gnb = GaussianNB()
    clf = GridSearchCV(gnb, param_grid).fit(X_train, y_train)
    print(clf)
    y_pred = clf.predict(X_test)

    return {'roc_auc_score': roc_auc_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)}

