from typing import Any, Dict, Tuple, Union

from gensim.models import FastText

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ExperimentClassifier = Union[LogisticRegression, GaussianNB, MultinomialNB, KNeighborsClassifier,
                             SVC, DecisionTreeClassifier]


def get_document(document_type: str) -> callable:
    document_types = {
        'bag_of_words': bag_of_words,
        'tf_idf': tf_idf,
        'word_embedding': word_embedding
    }

    try:
        return document_types[document_type]
    except KeyError:
        raise Exception(f'Unknown document type: {document_type}')


def get_model(model_type: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    model_types = {
        'decision_tree': {
            'model': {
                'bag_of_words': DecisionTreeClassifier,
                'tf_idf': DecisionTreeClassifier,
                'word_embedding': DecisionTreeClassifier
            },
            'hyperparameters': {
                'bag_of_words': {
                    'criterion': ['gini', 'entropy'],
                    'random_state': [0],
                },
                'tf_idf': {
                    'criterion': ['gini', 'entropy'],
                    'random_state': [0],
                },
                'word_embedding': {
                    'criterion': ['gini', 'entropy'],
                    'random_state': [0],
                }
            }
        },
        'naive_bayes': {
            'model': {
                'bag_of_words': MultinomialNB,
                'tf_idf': MultinomialNB,
                'word_embedding': GaussianNB,
            },
            'hyperparameters': {
                'bag_of_words': {
                    'alpha': np.logspace(2, -5, num=10)
                },
                'tf_idf': {
                    'alpha': np.logspace(2, -5, num=10)
                },
                'word_embedding': {
                    'var_smoothing': np.logspace(0, -15, num=10)
                }
            }
        },
        'knn': {
            'model': {
                'bag_of_words': KNeighborsClassifier,
                'tf_idf': KNeighborsClassifier,
                'word_embedding': KNeighborsClassifier
            },
            'hyperparameters': {
                'bag_of_words': {
                    'n_neighbors': list(range(3, 16, 2)),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'chebyshev']
                },
                'tf_idf': {
                    'n_neighbors': list(range(3, 16, 2)),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'chebyshev']
                },
                'word_embedding': {
                    'n_neighbors': list(range(3, 16, 2)),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'chebyshev']
                }
            }
        },
        'logistic_regression': {
            'model': {
                'bag_of_words': LogisticRegression,
                'tf_idf': LogisticRegression,
                'word_embedding': LogisticRegression
            },
            'hyperparameters': {
                'bag_of_words': {
                    'C': np.logspace(2, -5, num=10),
                    'solver': ['liblinear', 'sag', 'saga'],
                    'penalty': ['l2', 'l1'],
                    'max_iter': [10000]
                },
                'tf_idf': {
                    'C': np.logspace(2, -5, num=10),
                    'solver': ['liblinear', 'sag', 'saga'],
                    'penalty': ['l2', 'l1'],
                    'max_iter': [10000]
                },
                'word_embedding': {
                    'C': np.logspace(2, -5, num=10),
                    'solver': ['liblinear', 'sag', 'saga'],
                    'penalty': ['l2', 'l1'],
                    'max_iter': [10000]
                }
            }
        },
        'svm': {
            'model': {
                'bag_of_words': SVC,
                'tf_idf': SVC,
                'word_embedding': SVC
            },
            'hyperparameters': {
                'bag_of_words': {
                    'C': np.logspace(2, -5, num=10),
                    'kernel': ['linear', 'rbf', 'sigmoid', 'poly']
                },
                'tf_idf': {
                    'C': np.logspace(2, -5, num=10),
                    'kernel': ['linear', 'rbf', 'sigmoid', 'poly']
                },
                'word_embedding': {
                    'C': np.logspace(2, -5, num=10),
                    'kernel': ['linear', 'rbf', 'sigmoid', 'poly']
                }
            }
        }
    }

    try:
        return model_types[model_type]
    except KeyError:
        raise Exception(f'Unknown model type: {model_type}')


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
