import csv
import math
import sys
from collections import defaultdict
from pprint import pprint

import numpy
from tqdm import tqdm


# def get_all_attributes_amount(_attributes: list, _train_set: numpy.ndarray) -> list:
#     all_attributes_amount = []
#
#     for index in range(1, _train_set.shape[1]):
#         attrs, amounts = numpy.unique(_train_set[:, [0, index]], return_counts=True, axis=0)
#         all_attributes_amount.append(numpy.c_[attrs, amounts])
#
#     return all_attributes_amount


def calculate_entropy(_decision_attributes_amount: list) -> float:
    _attribute_entropy = 0
    for decision_attribute_amount in _decision_attributes_amount:
        _attribute_entropy -= decision_attribute_amount / numpy.sum(_decision_attributes_amount) * numpy.log2(
            decision_attribute_amount / numpy.sum(_decision_attributes_amount))

    return _attribute_entropy


def get_gains(_train_set: numpy.ndarray, set_information: float) -> [float]:
    gains = []

    for index in range(1, _train_set.shape[1]):
        unique, count = numpy.unique(_train_set[:, [0, index]], return_counts=True, axis=0)
        unique_attributes = numpy.unique(unique[:, 1])
        count_unique = numpy.c_[count, unique]
        grouped_count_unique = [count_unique[count_unique[:, 2] == unique_attribute] for unique_attribute in
                                unique_attributes]
        _attributes_entropy = 0.
        for group_amount in grouped_count_unique:
            amounts = [int(row[0]) for row in group_amount]
            _attributes_entropy += sum(amounts) / _train_set.shape[0] * calculate_entropy(amounts)
        gains.append(set_information - _attributes_entropy)

    return gains


def id3(_train_set: numpy.ndarray, _attributes: list) -> tuple:
    if numpy.all(_train_set[:, 0] == _train_set[:, 0][0]):
        return _train_set[:, 0][0]

    if len(_attributes) == 0:
        values, counts = numpy.unique(_train_set[:, 0], return_counts=True)
        return values[numpy.argmax(counts)]

    decision_attributes_amount = numpy.unique(_train_set[:, 0], return_counts=True)[1]
    decision_attributes_entropy = calculate_entropy(decision_attributes_amount)

    gains = get_gains(_train_set, decision_attributes_entropy)
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
                id3(new_train_set, new_attributes)
            ))

    return tree


def get_decision(_test_set_row: numpy.ndarray, _model: tuple) -> str:
    element_index = [element[0] for element in _model[1]].index(_test_set_row[_model[0]])
    new_model = _model[1][element_index][1]
    if isinstance(new_model, tuple):
        return get_decision(_test_set_row, _model[1][element_index][1])
    else:
        return new_model


def get_forest_decision(_test_set_row: numpy.ndarray, _trees: list) -> str:
    decisions = [get_decision(_test_set_row, tree) for tree in _trees]
    decision, amounts = numpy.unique(decisions, return_counts=True)

    return decision[numpy.argmax(amounts)]


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
            'acc': acc
        }

    return tables


def evaluate_forest(_test_set: numpy.ndarray, _trees: list) -> dict:
    results = [get_forest_decision(row, _trees) for row in _test_set[:, 1:]]

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
            'acc': acc
        }

    return tables


def load_data(file_name: str, file_split: str) -> (list, numpy.ndarray, numpy.ndarray):
    """
    funkcja zwraca atrybuty ze zbioru oraz train_set i test_set podzielony zgodnie z artybutem file_split.

    Atrybuty są listą gdzie pierwszy element to indeks używany później w id3, a drugi to grupa atrybutów z kolumny zbioru

    :param file_name: nazwa pliku
    :param file_split: sposób podziału. 1:0 - wszystko w pierwszy nic w drugi, 0:1 - wszystko w drugi nic w pierwszy
    :return: lista atrybutów bez klas, ndarray train_set, ndarray test_set
    """
    _file_split = [int(i) for i in file_split.split(':')]

    with open(file_name) as csvfile:
        whole_set = numpy.array([line for line in csv.reader(csvfile)])

    numpy.random.shuffle(whole_set)
    _attributes = [(col_index - 1, numpy.unique(whole_set[:, col_index])) for col_index in range(1, whole_set.shape[1])]
    _train_set = whole_set[:whole_set.shape[0] * _file_split[0] // sum(_file_split)]
    _test_set = whole_set[whole_set.shape[0] * _file_split[0] // sum(_file_split):]

    return _attributes, _train_set, _test_set


def create_forest(fit: callable, _attributes: list, _train_set: numpy.ndarray, _B: int) -> list:
    """
    funkcja tworzy (i szkoli) las

    :param fit: funkcja szkoląca przyjmuje argumenty train_set, attributes
    :param _attributes: zbiór atrybutów (bez klas)
    :param _train_set: zbiór trenujące
    :param _B: liczba elementów losowanych z _train_set oraz liczba utworzonych drzew
    :return: lista drzew utworzonych na podstawie fit oraz _B
    """
    _trees = []
    for _ in range(_B):
        _train_set_copy = _train_set.copy()
        _attributes_copy = _attributes[:]

        #  losowe indeksy atrybutów bez zwracania
        random_attributes_indexes = numpy.random.choice(len(_attributes_copy),
                                                        math.floor(math.sqrt(len(_attributes_copy))),
                                                        replace=False)
        #  losowe indeksy zbioru trenującego ze zwracaniem
        random_train_indexes = numpy.random.choice(_train_set_copy.shape[0], _B)

        #  tworzy set z indeksów
        _train_set_copy = _train_set_copy[random_train_indexes]
        #  wywala atrybuty, które zostały usunięte przy losowaniu
        _train_set_copy = _train_set_copy[:, [0, *(numpy.sort(random_attributes_indexes) + 1)]]

        #  sortuje indeksy atrybutów i tworzy z nich set
        _attributes_copy = [_attributes_copy[index] for index in numpy.sort(random_attributes_indexes)]

        _trees.append(fit(_train_set_copy, _attributes_copy))

    return _trees


def cross_validation(fit: callable, validate: callable, _attributes: list, _train_set: numpy.ndarray, _cv: int):
    """
    Funkcja odpowiedzialna za przeprowadzenie cross-val. Wyświetla wyniki RMSE na podstawie tabeli błędów - TP, FP, FN, TN

    :param fit: funkcja szkoląca przyjmuje argumenty train_set, attributes
    :param validate: funkcja walidująca przyjmuje argumenty test_set, model
    :param _attributes: zbiór atrybutów (bez klas)
    :param _train_set: zbiór trenujące
    :param _cv: liczba podziałów
    """
    _splitted = numpy.array_split(train_set, _cv)  # rozbicie setu na cv podlist

    # wyniki w strukturze:
    # {
    #   'klasa1':
    #   {
    #       'acc': [],
    #       'rec': [],
    #       'prec': []
    #    },
    #    'klasa2':
    #    {
    #       'acc': [],
    #       'rec': [],
    #       'prec': []
    #    },
    #     ...
    # }
    _results = defaultdict(lambda: defaultdict(list))  # workaround, bo pythong ssie >:(
    for key in _train_set[:, 0]:
        _results[key]['acc'] = []
        _results[key]['rec'] = []
        _results[key]['prec'] = []

    for index in tqdm(range(_cv)):
        _splitted_copy = _splitted[:]
        _test_set = _splitted_copy.pop(index)  # U_i
        _new_train_set = numpy.concatenate(_splitted_copy)  # U - U_i

        _model = fit(_new_train_set, _attributes)
        _result = validate(_test_set, _model)

        for key in _result:
            _results[key]['acc'].append(_result[key]['acc'])
            _results[key]['rec'].append(_result[key]['rec'])
            _results[key]['prec'].append(_result[key]['prec'])

    # wyniki są przedstawiane jako RMSE od dokładności, precyzji i recall
    for key in _results:  # potencjalnie zmienić, jesli użyjemy numby
        print(key)
        print('acc:', numpy.linalg.norm(numpy.array(_results[key]['acc']) - 1) / numpy.sqrt(len(_results[key]['acc'])))
        print('rec:', numpy.linalg.norm(numpy.array(_results[key]['rec']) - 1) / numpy.sqrt(len(_results[key]['rec'])))
        print('prec:',
              numpy.linalg.norm(numpy.array(_results[key]['prec']) - 1) / numpy.sqrt(len(_results[key]['prec'])))


if __name__ == '__main__':
    # szkolenie drzewa i ewaluacja
    attributes, train_set, test_set = load_data(sys.argv[1], '3:2')
    model = id3(train_set, attributes)
    evaluation = evaluate(test_set, model)
    print('id3')
    pprint(evaluation)

    print()

    # szkolenie lasu drzew i ewaluacja
    attributes, train_set, test_set = load_data(sys.argv[1], '3:2')
    trees = create_forest(id3, attributes, train_set, 150)
    evaluation = evaluate_forest(test_set, trees)
    print('id3 forest')
    pprint(evaluation)

    print()

    # cross-val drzewa
    attributes, train_set, test_set = load_data(sys.argv[1], '3:2')
    cross_validation(id3, evaluate, attributes, train_set, 5)

    print()

    # 1:0 zwraca pełen train_set i pusty test_set. 0:1 zwraca pusty train_set i pełen test_set
    attributes, train_set, _ = load_data(sys.argv[1], '1:0')
    # cross-val lasu
    cross_validation(
        lambda train_set, attributes: create_forest(id3, attributes, train_set, 150),
        evaluate_forest,
        attributes,
        train_set,
        5
    )
