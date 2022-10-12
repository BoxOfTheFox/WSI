from tqdm import tqdm

from id3 import *
from bayes import *

import numpy

if __name__ == '__main__':
    with open("tables.json") as json_file:
        tables = json.load(json_file)

    with open("network.json") as json_file:
        network = json.load(json_file)

    for iters in [100, 1_000, 10_000, 100_000]:
        print(iters)
        evals = []
        for _ in range(25):
            states, root_name = load_data(network, tables)
            answers = [get_answers(states, root_name) for _ in range(iters)]

            with open('back.csv', 'w', newline='') as csvfile:
                fieldnames = ['ache', 'back', 'sport', 'chair']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(answers)

            attributes, train_set, test_set = load_data_id3("back.csv", '3:2')
            model = fit(train_set, attributes)
            evals.append(evaluate(test_set, model))

        for val1 in ['False', 'True']:
            print(val1)
            for val2 in ['prec', 'rec', 'acc', 'tp', 'fp', 'fn', 'tn']:
                lst = list(map(lambda dct: dct[val1][val2], evals))
                print(f'{val2} {numpy.mean(lst):.2f}~{numpy.std(lst)/numpy.mean(lst):.4f}% ↓{numpy.min(lst):.2f} ↑{numpy.max(lst):.2f}')
            print('-'*30)
