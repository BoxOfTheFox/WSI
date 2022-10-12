from typing import Callable

import cec2017.functions
import numba
import numpy

N = 10_000
BOUND = 100
DIMEN = 10


def _rate(q: Callable, population: numpy.ndarray) -> numpy.ndarray:
    return numpy.array(list(map(q, population)))


# @numba.njit
def _find_best(population: numpy.ndarray, population_rates: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
    best_specimen_index = numpy.argmin(population_rates)
    return population[best_specimen_index], population_rates[best_specimen_index]


@numba.njit
def _reproduction(
        population: numpy.ndarray,
        population_rates: numpy.ndarray,
        new_population_size: int
) -> numpy.ndarray:
    new_population = numpy.empty((new_population_size, DIMEN))

    for index in range(new_population_size):
        specimen_indexes = numpy.random.choice(population.shape[0], (1, 2))[0]
        specimen_rates = numpy.array([population_rates[index] for index in specimen_indexes])
        better_specimen_index = specimen_indexes[numpy.argmin(specimen_rates)]
        new_population[index] = population[better_specimen_index]

    return new_population


# @numba.njit
def _mutation(population: numpy.ndarray, mutation_strength) -> numpy.ndarray:
    for specimen in population:
        specimen += mutation_strength * numpy.random.randn(DIMEN)

    return population

#nth best
# @numba.njit
def _succession(
        population: numpy.ndarray,
        new_population: numpy.ndarray,
        population_rates: numpy.ndarray,
        new_population_rates: numpy.ndarray,
        elite: int
) -> (numpy.ndarray, numpy.ndarray):
    population_sorted_rates = population_rates.argsort()
    population = population[population_sorted_rates]
    population_rates = population_rates[population_sorted_rates]

    return_population = numpy.concatenate((population[:elite], new_population))
    return_population_rates = numpy.concatenate((population_rates[:elite], new_population_rates))

    return_population_sorted_rates = return_population_rates.argsort()
    return_population = return_population[return_population_sorted_rates]
    return_population_rates = return_population_rates[return_population_sorted_rates]

    return return_population[:-elite], return_population_rates[:-elite]


def evolution(
        q=cec2017.functions.f4,
        population_size=2,
        mutation_strength=0.5,
        elite=2
) -> (numpy.ndarray, numpy.ndarray):
    population = numpy.random.uniform(-BOUND, BOUND, size=(population_size, DIMEN))
    population_rates = _rate(q, population)
    best_specimen, best_specimen_rating = _find_best(population, population_rates)

    for cycle in range(N//population_size-1):
        new_population = _reproduction(population, population_rates, population.shape[0])
        new_population = _mutation(new_population, mutation_strength)
        new_population_rates = _rate(q, new_population)
        best_new_specimen, best_new_specimen_rating = _find_best(new_population, new_population_rates)

        if best_new_specimen_rating <= best_specimen_rating:
            # print(cycle)
            # print(best_new_specimen_rating, best_specimen_rating)
            best_specimen_rating = best_new_specimen_rating
            best_specimen = best_new_specimen

        population, population_rates = _succession(population, new_population, population_rates, new_population_rates,
                                                   elite)

    return best_specimen, best_specimen_rating


if __name__ == '__main__':
    specimen, specimen_rates = evolution()
    print(specimen)
    print(specimen_rates)
