import numpy.random
from matplotlib import pyplot as plt
from autograd import grad


def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def stop(old_x, new_x, q, value):
    return q(old_x) - q(new_x) > value


def descent(upper_bound=100, stop_step=0.001, step=0.01, dimensionality=2, q=booth, debug_step=100):
    x = numpy.random.uniform(-upper_bound, upper_bound, size=dimensionality)
    old_x = x * 2

    if debug_step != -1:
        last_q = q(x)
        print(f'q(x) = {q(x)}')

    debug = debug_step

    while stop(old_x, x, q, stop_step):
        old_x = x.copy()
        grad_fct = grad(q)
        gradient = grad_fct(x)
        x -= step * gradient
        if debug == 0:
            print(f'q(x) = {q(x)}, delta={last_q-q(x)}')
            last_q = q(x)
            debug = debug_step
        else:
            debug -= 1
        plt.arrow(old_x[0], old_x[1], x[0] - old_x[0], x[1] - old_x[1], head_width=1, head_length=1, fc='k', ec='k')

    if debug_step != -1:
        print("found spot:", x)
        print(f'q(x) = {q(x)}')
        print("Preparing plot")

    x = numpy.arange(-upper_bound, upper_bound, 0.1)
    y = numpy.arange(-upper_bound, upper_bound, 0.1)
    z = q(numpy.meshgrid(x, y, sparse=True))

    if debug_step != -1:
        print("Plotting")

    plt.contour(x, y, z, 50)
    plt.show()


descent()
# descent(step=0.00000000135, dimensionality=10, q=cec2017.functions.f1)
# descent(step=0.00000000000000000095, stop_step=1_000_000_000_000, dimensionality=10, q=cec2017.functions.f2)
# descent(step=0.0000000057, stop_step=1, dimensionality=10, q=cec2017.functions.f3)
