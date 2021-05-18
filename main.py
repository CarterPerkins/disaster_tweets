import pandas as pd

from experiment import run_experiment
from preprocessing import preprocess

if __name__ == '__main__':
    df = preprocess()
    run_experiment(df)
