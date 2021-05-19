from typing import Tuple

import numpy as np
import pandas as pd


def split_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df['text'].to_numpy()
    y = df['target'].to_numpy()
    return (X, y)
