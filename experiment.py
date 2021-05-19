from typing import Dict, Tuple

import numpy as np
import pandas as pd

from naive_bayes import run_naive_bayes


def run_experiment(df: pd.DataFrame, name: str, settings: Dict) -> Dict:
    callbacks = {
        'naive_bayes': lambda: run_naive_bayes(df, **settings)
    }

    return callbacks[name]()

'''
        train_df, val_test_df = train_test_split(df, test_size=0.3,
                                                 random_state=0)
        val_df, test_df = train_test_split(val_test_df, test_size=0.666,
                                           random_state=0)
        X_train, y_train = split_dataframe(train_df)
        X_val, y_val = split_dataframe(val_df)
        X_test, y_test = split_dataframe(test_df)
'''
