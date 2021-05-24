from typing import Tuple

from gensim.models import FastText

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def split_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df['text'].to_numpy()
    y = df['target'].to_numpy()
    return (X, y)

def bag_of_words(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Naive Bayes with Bag of Words
    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train).toarray()
    X_val_counts = cv.transform(X_val).toarray()
    X_test_counts = cv.transform(X_test).toarray()

    return (X_train_counts, X_val_counts, X_test_counts)

def tf_idf(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Naive Bayes with TF-IDF Vectors
    tv = TfidfVectorizer()
    X_train_tfidf = tv.fit_transform(X_train).toarray()
    X_val_tfidf = tv.transform(X_val).toarray()
    X_test_tfidf = tv.transform(X_test).toarray()

    return (X_train_tfidf, X_val_tfidf, X_test_tfidf)

def word_embedding(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.DataFrame(X_train)[0]
    val_df = pd.DataFrame(X_val)[0]
    test_df = pd.DataFrame(X_test)[0]
    corpus = pd.concat([train_df, val_df, test_df]).apply(lambda x: x.split())
    ft = FastText(corpus)
    # convert to word embeddings
    train_df = train_df.apply(lambda x: [ft.wv[k] for k in x.split()])
    val_df = val_df.apply(lambda x: [ft.wv[k] for k in x.split()])
    test_df = test_df.apply(lambda x: [ft.wv[k] for k in x.split()])

    # average word embeddings
    train_df = train_df.apply(lambda x: np.mean(x, axis=0))
    val_df = val_df.apply(lambda x: np.mean(x, axis=0))
    test_df = test_df.apply(lambda x: np.mean(x, axis=0))

    X_train_avg_we = np.stack(train_df.to_numpy())
    X_val_avg_we = np.stack(val_df.to_numpy())
    X_test_avg_we = np.stack(test_df.to_numpy())

    return (X_train_avg_we, X_val_avg_we, X_test_avg_we)
