import random

import numba
import numpy


@numba.jit(nopython=True)
def core(W, p, w):
    best_value = 0
    best_space = 0
    best_weight = 0

    for i in range(len(p)):
        if best_weight + w[i] < W:
            best_value += p[i]
            best_weight += w[i]
            best_space += 2**i

    return best_space, best_value, best_weight


def brute_fast(
        w=numpy.array([8, 3, 5, 2]),  # waga przedmiotów
        W=9,  # maksymalna waga plecaka
        p=numpy.array([16, 8, 9, 6]),  # wartość przedmiotów
        sort_type=None
) -> (int, int, int):
    if 'inc' == sort_type:
        array_mapping = (p / w).argsort()
    elif 'dec' == sort_type:
        array_mapping = (-p / w).argsort()
    else:
        array_mapping = numpy.arange(0, len(w))

    w = w[array_mapping]
    p = p[array_mapping]

    best_space, best_value, best_weight = core(W, p, w)

    return bin(best_space), best_value, best_weight


if __name__ == '__main__':
    random.seed()
    numpy.random.seed()

    w = numpy.random.randint(1, 100, 250_000_000)  # waga przedmiotów
    p = numpy.random.randint(1, 100, 250_000_000)  # wartość przedmiotów

    print(brute_fast(w=w, W=int(numpy.mean(w) * 6), p=p, sort_type='dec'))
