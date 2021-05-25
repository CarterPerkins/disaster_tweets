import json

from experiment import run_experiment
from preprocessing import preprocess


if __name__ == '__main__':
    partitions = preprocess()
    for df in partitions:
        print(df.head())

    experiments = []
    for model in ['naive_bayes', 'knn', 'logistic_regression', 'svm', 'neural_network']:
        if model not in ['neural_network']:
            continue
        for document in ['bag_of_words', 'tf_idf', 'word_embedding']:
            results = run_experiment(partitions, document, model)
            experiments.append(results)
            print(results)
            print('-'*80)

    with open('output.json', 'w', encoding='utf-8') as fp:
        json.dump(experiments, fp, ensure_ascii=False, indent=4)
