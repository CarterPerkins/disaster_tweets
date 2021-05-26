import time
from typing import Dict, Tuple

import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from utility import get_document, get_model, split_dataframe


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
    clf = get_model(model)['model'][document_type]()
    settings = get_model(model)['hyperparameters'][document_type]
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

    test_end = time.time()
    print(f'\tTested in {test_end - test_start:.3f}s')
    elapsed = test_end - partition_start
    print(f'Experiment finished in {elapsed:.3f}s')

    results = {
        'roc_auc_score': roc_auc_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'accuracy_score': accuracy_score(y_test, y_pred),
        'precision_score': precision_score(y_test, y_pred),
        'recall_score': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'model': model,
        'document_type': document_type,
        'elapsed': elapsed,
        'params': search.best_params_
    }

    return results
