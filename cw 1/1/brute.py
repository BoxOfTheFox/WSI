import numpy


def bits(n: int):
    while n:
        b = n & (~n + 1)
        yield int(numpy.log2(b))
        n ^= b


def brute(
        w=numpy.array([8, 3, 5, 2]),  # waga ModalDiaprzedmiotów
        W=9,  # maksymalna waga plecaka
        p=numpy.array([16, 8, 9, 6]),  # wartość przedmiotów
        sort_type=None,
        debug=False
) -> (int, int, int):
    best_value = 0
    best_space = 0
    best_weight = 0

    if 'inc' == sort_type:
        array_mapping = (p / w).argsort()
    elif 'dec' == sort_type:
        array_mapping = (-p / w).argsort()
    else:
        array_mapping = numpy.arange(0, len(w))

    w = w[array_mapping]
    p = p[array_mapping]

    if debug:
        print('weights:', w)
        print("values:", p)
        print("max weight:", W)

    for space in range(2 ** len(w)):
        new_weight = 0
        new_value = 0

        for position in bits(space):
            new_weight += w[position]
            if W < new_weight:
                break
            new_value += p[position]

        if W >= new_weight and best_value < new_value:
            best_value = new_value
            best_space = space
            best_weight = new_weight

    if debug:
        print(bin(best_space), best_value, best_weight)

    return bin(best_space), best_value, best_weight


if __name__ == '__main__':
    brute(debug=True, sort_type=None)
    brute(debug=True, sort_type='inc')
    brute(debug=True, sort_type='dec')
