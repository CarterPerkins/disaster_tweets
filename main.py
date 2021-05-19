import pandas as pd

from experiment import run_experiment
from preprocessing import preprocess


if __name__ == '__main__':
    df = preprocess()
    for mode in ['tf_idf', 'bag_of_words']:
        results = run_experiment(df, 'naive_bayes',
                                 {'mode': mode,
                                  'param_grid': {'var_smoothing': [pow(10, -k) for k in range(20)]}})
        print(mode, results)
