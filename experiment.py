import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from utility import bag_of_words, tf_idf, split_dataframe, word_embedding


def get_document(document_type: str):
    document_types = {
        'bag_of_words': bag_of_words,
        'tf_idf': tf_idf,
        'word_embedding': word_embedding
    }

    try:
        return document_types[document_type]
    except KeyError:
        raise Exception(f'Unknown document type: {document_type}')

def get_model(model_type: str):
    model_types = {
        'naive_bayes': {
            'model': GaussianNB,
            'hyperparameters': {
                'var_smoothing': np.logspace(0, -10, num=100)
            }
        },
        'knn': {
            'model': KNeighborsClassifier,
            'hyperparameters': {
                'n_neighbors': list(range(3, 14, 2)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'chebyshev']
            }
        },
        'logistic_regression': {
            'model': LogisticRegression,
            'hyperparameters': {
                'C': list(10**k for k in range(2, -11, -1)),
                'solver': ['liblinear', 'sag', 'saga'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'max_iter': [10000]
            }
        }
    }

    try:
        return model_types[model_type]
    except KeyError:
        raise Exception(f'Unknown model type: {model_type}')

def run_experiment(partitions: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                   document_type: str,
                   model: str) -> Dict[str, float]:
    print('Starting experiment...')
    # Setup partitions
    partition_start = time.time()
    train_df, val_df, test_df = partitions
    X_train, y_train = split_dataframe(train_df)
    X_val, y_val = split_dataframe(val_df)
    X_test, y_test = split_dataframe(test_df)
    doc_transform = get_document(document_type)
    X_train, X_val, X_test = doc_transform(X_train, X_val, X_test)
    partition_end = time.time()
    print(f'\tConverted documents in {partition_end - partition_start:.3f}s')
    # Train the model
    train_start = time.time()
    clf, settings = get_model(model)['model'](), get_model(model)['hyperparameters']
    trained_model = clf.fit(X_train, y_train)
    train_end = time.time()
    print(f'\tTrained in {train_end - train_start:.3f}s')

    # Tune the model
    tune_start = time.time()
    search = GridSearchCV(trained_model, settings, scoring='f1')
    tuned_model = search.fit(X_val, y_val)
    tune_end = time.time()
    print(f'\tTuned in {tune_end - tune_start:.3f}s')

    # Test the model
    test_start = time.time()
    y_pred = tuned_model.predict(X_test)
    results = {
        'roc_auc_score': roc_auc_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'accuracy_score': accuracy_score(y_test, y_pred),
        'model': model,
        'document_type': document_type,
    }
    test_end = time.time()
    print(f'\tTested in {test_end - test_start:.3f}s')
    print(confusion_matrix(y_test, y_pred))
    print(f'Experiment finished in {test_end - partition_start:.3f}s')

    return results
