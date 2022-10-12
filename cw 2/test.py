from tqdm import tqdm

import cec2017.functions
import numpy

from cw3.evolution import evolution


def _evaluate(
        q=cec2017.functions.f4,
        population_size=2,
        mutation_strength=0.3,
        elite=2
) -> (float, float, float, float):
    rates = [evolution(q, population_size, mutation_strength, elite)[1] for _ in range(25)]
    return min(rates), numpy.average(rates), numpy.std(rates), max(rates)


def _print_table(_scores: [], _parameter: str, _values: []):
    print(" ----------------------------------------")
    print("|Parametr|  min  |  śr   |  std  |  max  |")
    print(" ----------------------------------------")
    for _index, _score in enumerate(_scores):
        value_string = f"{_values[_index]:.2f}" if type(_values[_index]) is numpy.float64 else f"{_values[_index]}"
        print(f"|{_parameter}={value_string}",
              ' ' * (len("Parametr") - len(_parameter) - len(value_string) - 1),
              sep='', end='')
        print(f"|{_score[0]:.2f}", ' ' * (7 - len(f'{_score[0]:.2f}')), sep='', end='')
        print(f"|{_score[1]:.2f}", ' ' * (7 - len(f'{_score[1]:.2f}')), sep='', end='')
        print(f"|{_score[2]:.2f}", ' ' * (7 - len(f'{_score[2]:.2f}')), sep='', end='')
        print(f"|{_score[3]:.2f}", ' ' * (7 - len(f'{_score[3]:.2f}')), '|', sep='')
    print(" ----------------------------------------")


def _test(q=cec2017.functions.f4):
    scores = []
    # population_sizes = [*range(2, 10), *range(10, 100, 10), *range(100, 1000, 100), *range(1000, 10000, 1000), 10000]
    # population_sizes = [*range(2, 5)]
    population_sizes = [50,200,7000]
    # for population_size in tqdm(population_sizes):
    #     scores.append(_evaluate(q=q, population_size=population_size))
    # best_avg_population_size_index = numpy.array(scores)[:, 1].argmin()
    # best_min_population_size_index = numpy.array(scores)[:, 0].argmin()
    # print()
    # _print_table(scores, "mu", population_sizes)
    # print(f'\n~~~population size from avg: {population_sizes[best_avg_population_size_index]}~~~\n')
    # print(f'\n~~~population size from min: {population_sizes[best_min_population_size_index]}~~~\n')

    # scores = []
    # elite_sizes = range(1, population_sizes[best_avg_population_size_index] + 1)
    elite_sizes = [14,19,2600]
    # for elite_size in tqdm(elite_sizes):
    #     scores.append(_evaluate(q=q, population_size=population_sizes[best_avg_population_size_index], elite=elite_size))
    # best_avg_population_elite_index = numpy.array(scores)[:, 1].argmin()
    # print()
    # _print_table(scores, "k", elite_sizes)
    # print(f'\n~~~elite size from avg: {elite_sizes[best_avg_population_elite_index]}~~~\n')

    # scores = []
    # elite_sizes = range(1, population_sizes[best_min_population_size_index] + 1)
    # for elite_size in tqdm(elite_sizes):
    #     scores.append(
    #         _evaluate(q=q, population_size=population_sizes[best_min_population_size_index], elite=elite_size))
    # best_min_population_elite_index = numpy.array(scores)[:, 0].argmin()
    # print()
    # _print_table(scores, "k", elite_sizes)
    # print(f'\n~~~elite size from min: {elite_sizes[best_min_population_elite_index]}~~~\n')

    scores = []
    mutation_strengths = numpy.arange(1.1, 5.1, 0.1)
    for mutation_strength in tqdm(mutation_strengths):
        scores.append(_evaluate(q=q,
                                population_size=population_sizes[1],
                                elite=elite_sizes[1],
                                mutation_strength=mutation_strength))
    best_avg_population_mutation_strength_index = numpy.array(scores)[:, 1].argmin()
    print()
    _print_table(scores, "s", mutation_strengths)
    print(f'\n~~~mutation strength from avg: {mutation_strengths[best_avg_population_mutation_strength_index]}~~~\n')

    scores = []
    for mutation_strength in tqdm(mutation_strengths):
        scores.append(_evaluate(q=q,
                                population_size=population_sizes[0],
                                elite=elite_sizes[0],
                                mutation_strength=mutation_strength))
    best_min_population_mutation_strength_index = numpy.array(scores)[:, 1].argmin()
    print()
    _print_table(scores, "s", mutation_strengths)
    print(f'\n~~~mutation strength from min: {mutation_strengths[best_min_population_mutation_strength_index]}~~~\n')

    scores = []
    for mutation_strength in tqdm(mutation_strengths):
        scores.append(_evaluate(q=q,
                                population_size=population_sizes[2],
                                elite=elite_sizes[2],
                                mutation_strength=mutation_strength))
    best_min_population_mutation_strength_index = numpy.array(scores)[:, 2].argmin()
    print()
    _print_table(scores, "s", mutation_strengths)
    # print(f'\n~~~mutation strength from min: {mutation_strengths[best_min_population_mutation_strength_index]}~~~\n')


if __name__ == '__main__':
    # _test()
    _test(cec2017.functions.f5)
