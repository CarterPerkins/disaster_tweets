'''
Author: Carter Perkins
Last Updated: 10 June 2021
'''
from functools import partial
from multiprocessing import Pool
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
                            recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

from utility import get_document, get_model, split_dataframe, ExperimentClassifier


def train_and_validate(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray,
                       y_val: np.ndarray, clf: ExperimentClassifier,
                       params: Dict[str, Any]):

    '''

    '''
    start = time.time()
    try:
        model = clf(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
    except:
        f1 = -1
    elapsed = time.time() - start
    print(f'\t\tElapsed: {elapsed:.3f}s\tF1: {f1:.3f}')
    return {'scorer': f1_score(y_val, y_pred), 'model': model, 'params': params}


def exhaustive_train(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray,
                     y_val: np.ndarray, clf: ExperimentClassifier, param_grid: Dict[str, Any],
                     processes: int):
    '''

    '''

    grid_search = ParameterGrid(param_grid)

    with Pool(processes=processes) as pool:
        func_wrapper = partial(train_and_validate, X_train, X_val, y_train, y_val, clf)
        results = pool.map_async(func_wrapper, grid_search)
        results = results.get()

    return results


def run_experiment(partitions: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                   document_type: str,
                   model: str,
                   n_jobs: int) -> Dict[str, Any]:
    '''

    '''
    print('Starting experiment...')
    print(f'\tModel: {model}\t\tDocument Type: {document_type}')
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
    clf = get_model(model)['model'][document_type]
    param_grid = get_model(model)['hyperparameters'][document_type]
    grid_results = exhaustive_train(X_train, X_val, y_train, y_val, clf, param_grid, n_jobs)
    results = max(grid_results, key=lambda x: x['scorer'])
    train_end = time.time()
    print(f'\tTrained in {train_end - train_start:.3f}s')

    # Test the model
    test_start = time.time()
    tuned_model = results['model']
    y_pred = tuned_model.predict(X_test)

    test_end = time.time()
    print(f'\tTested in {test_end - test_start:.3f}s')
    elapsed = test_end - partition_start
    print(f'Experiment finished in {elapsed:.3f}s')

    payload = {
        'results': {
            'roc_auc_score': roc_auc_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'accuracy_score': accuracy_score(y_test, y_pred),
            'precision_score': precision_score(y_test, y_pred),
            'recall_score': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'model': model,
            'document_type': document_type,
            'elapsed': elapsed,
            'best_params': results['params'],
            'grid_results': grid_results
        }
    }

    return payload
