import json

from experiment import run_experiment
from preprocessing import preprocess


if __name__ == '__main__':
    partitions = preprocess()
    for df in partitions:
        print(df.head())

    experiments = []
    for model in ['naive_bayes', 'knn', 'logistic_regression', 'svm', 'decision_tree','neural_network']:
        if model in ['neural_network']:
            continue
        for document in ['bag_of_words', 'tf_idf', 'word_embedding']:
            payload = run_experiment(partitions, document, model, None)
            experiments.append(payload['results'])

            print('f1\t\tacc\t\troc_auc')
            print(payload['results']['f1_score'],
                  payload['results']['accuracy_score'],
                  payload['results']['roc_auc_score'])
            print('-'*80)

    with open('output.json', 'w', encoding='utf-8') as fp:
        json.dump(experiments, fp, ensure_ascii=False, indent=4)
