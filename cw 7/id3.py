import csv
import sys
from pprint import pprint

import numpy


def get_all_attributes_amount(_attributes: list, _train_set: numpy.ndarray) -> list:
    all_attributes_amount = []

    for index in range(1, _train_set.shape[1]):
        attrs, amounts = numpy.unique(_train_set[:, [0, index]], return_counts=True, axis=0)
        all_attributes_amount.append(numpy.c_[attrs, amounts])

    return all_attributes_amount


def calculate_entropy(_decision_attributes_amount: list) -> float:
    _attribute_entropy = 0
    for decision_attribute_amount in _decision_attributes_amount:
        _attribute_entropy -= decision_attribute_amount / numpy.sum(_decision_attributes_amount) * numpy.log2(
            decision_attribute_amount / numpy.sum(_decision_attributes_amount))

    return _attribute_entropy


def calculate_set_entropy(_attributes_amount: numpy.ndarray, _attributes: list, _examples_amount: int) -> float:
    _attributes_entropy = 0.
    for attribute in _attributes:
        amounts = _attributes_amount[_attributes_amount[:, 1] == attribute][:, 2].astype(int)
        _attributes_entropy += numpy.sum(amounts) / _examples_amount * calculate_entropy(amounts)

    return _attributes_entropy


def fit(_train_set: numpy.ndarray, _attributes: list) -> tuple:
    if numpy.all(_train_set[:, 0] == _train_set[:, 0][0]):
        return _train_set[:, 0][0]

    if len(_attributes) == 0:
        values, counts = numpy.unique(_train_set[:, 0], return_counts=True)
        return values[numpy.argmax(counts)]

    decision_attributes_amount = numpy.unique(_train_set[:, 0], return_counts=True)[1]
    decision_attributes_entropy = calculate_entropy(decision_attributes_amount)

    gains = []
    attributes_amount = get_all_attributes_amount(_attributes, _train_set)
    for index, attribute in enumerate([row[1] for row in _attributes]):
        gains.append(decision_attributes_entropy - calculate_set_entropy(attributes_amount[index], attribute,
                                                                         _train_set.shape[0]))

    attribute_index = numpy.argmax(gains)
    tree = (_attributes[attribute_index][0], [])

    new_attributes = _attributes[:]
    del new_attributes[attribute_index]

    for attribute in _attributes[attribute_index][1]:
        new_train_set = _train_set[_train_set[:, attribute_index + 1] == attribute]
        new_train_set = numpy.delete(new_train_set, attribute_index + 1, 1)
        if new_train_set[:, 0].shape[0] == 0:
            values, counts = numpy.unique(_train_set[:, 0], return_counts=True)
            tree[1].append((attribute, values[numpy.argmax(counts)]))
        else:
            tree[1].append((
                attribute,
                fit(new_train_set, new_attributes)
            ))

    return tree


def get_decision(_test_set_row: numpy.ndarray, _model: tuple) -> str:
    element_index = [element[0] for element in _model[1]].index(_test_set_row[_model[0]])
    new_model = _model[1][element_index][1]
    if isinstance(new_model, tuple):
        return get_decision(_test_set_row, _model[1][element_index][1])
    else:
        return new_model


def evaluate(_test_set: numpy.ndarray, _model: tuple) -> dict:
    results = [get_decision(row, _model) for row in _test_set[:, 1:]]

    tables = {}

    for true_positive in numpy.unique(_test_set[:, 0]):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for true_result, model_result in zip(_test_set[:, 0], results):
            tp += true_positive == true_result == model_result
            fp += true_positive == model_result and true_result != true_positive
            fn += true_positive != model_result and true_result == true_positive
            tn += true_positive != model_result and true_result != true_positive
        # print(true_positive)
        # print(f'tp={tp} fp={fp} fn={fn} tn={tn}')
        # print(f'Prec={tp/(tp+fp)} Rec={tp/(tp+fn)} Acc={(tp + tn)/(tp+fp+fn+tn)}')

        if tp == fp == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)

        if tp == fn == 0:
            rec = 0
        else:
            rec = tp / (tp + fn)

        if tp == fp == fn == tn == 0:
            acc = 0
        else:
            acc = (tp + tn) / (tp + fp + fn + tn)

        tables[true_positive] = {
            'prec': prec,
            'rec': rec,
            'acc': acc,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

    return tables


def load_data_id3(file_name: str, file_split: str) -> (list, numpy.ndarray, numpy.ndarray):
    _file_split = [int(i) for i in file_split.split(':')]

    with open(file_name) as csvfile:
        whole_set = numpy.array([line for line in csv.reader(csvfile)])

    numpy.random.shuffle(whole_set)
    _attributes = [(col_index - 1, numpy.unique(whole_set[:, col_index])) for col_index in range(1, whole_set.shape[1])]
    _train_set = whole_set[:whole_set.shape[0] * _file_split[0] // sum(_file_split)]
    _test_set = whole_set[whole_set.shape[0] * _file_split[0] // sum(_file_split):]

    return _attributes, _train_set, _test_set


if __name__ == '__main__':
    attributes, train_set, test_set = load_data_id3(sys.argv[1], '3:2')

    model = fit(train_set, attributes)
    pprint(model)

    evaluate(test_set, model)
