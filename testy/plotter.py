import re
from pprint import pprint

import numpy
import matplotlib.pyplot as plt


def min_to_s(line: str) -> str:

    if 'min' in line:
        lst = list(map(int, re.findall('\d+', line)))
        line = str(int(lst[0])*60 + (int(lst[1]) if len(lst) > 1 is not None else 0)) + " s"

    return line


def split_line(line: str) -> ({float, str}, {float, str}):
    line1 = line.strip("\n").split('±')
    line2 = line1[1].split('per')

    line1split = min_to_s(line1[0]).strip(' ').split(' ')
    value1 = line1split[0]
    unit1 = line1split[1]

    line2split = line2[0].strip(' ').split(' ')
    value2 = line2split[0]
    unit2 = line2split[1]

    return {'value': float(value1), 'unit': unit1}, {'value': float(value2), 'unit': unit2}


def _normalize_to_ns(data: {float, str}) -> float:
    return_value = data['value']
    if data['unit'] == 'µs':
        return_value *= 1_000
    elif data['unit'] == 'ms':
        return_value *= 1_000_000
    elif data['unit'] == 's':
        return_value *= 1_000_000_000
    return return_value


def normalize_to_ns(data: ({float, str}, {float, str})) -> (float, float):
    return _normalize_to_ns(data[0]), _normalize_to_ns(data[1])


def draw_error_plot(lst: list, title: str):
    # construct some data like what you have:
    mins = numpy.asarray(list(map(lambda l: l['min'][0] / 1_000_000_000, lst)))
    maxes = numpy.asarray(list(map(lambda l: l['max'][0] / 1_000_000_000, lst)))
    means = numpy.asarray(list(map(lambda l: l['mean'][0] / 1_000_000_000, lst)))
    std = numpy.asarray(list(map(lambda l: l['std'][0] / 1_000_000_000, lst)))
    # create stacked errorbars:
    plt.errorbar(numpy.arange(4, mins.size + 4), means, std)
    plt.errorbar(numpy.arange(4, mins.size + 4), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='gray', lw=1)
    # plot and axis names
    plt.xlabel("Ilość przedmiotów")
    plt.ylabel("czas [s]")
    plt.title(title)
    plt.show()


def draw_plots_with_labels(plots: [{list, str}], title: str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for plot in plots:
        means = numpy.asarray(list(map(lambda l: l['mean'][0] / 1_000_000_000, plot['list'])))
        ax.plot(numpy.arange(4, means.size + 4), means, label=plot['label'])

    ax.legend()
    plt.xlabel("Ilość przedmiotów")
    plt.ylabel("czas [s]")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    with(open('testy/all.txt', 'r')) as reader:
        lines = reader.readlines()

    type_none = []
    type_inc = []
    type_dec = []

    type_nones = []
    type_incs = []
    type_decs = []

    for index in range(1, len(lines), 4):
        if lines[index - 1].strip('\n').split(' ')[1] == '0':
            type_none = numpy.array(type_none)
            type_inc = numpy.array(type_inc)
            type_dec = numpy.array(type_dec)

            if type_none.size > 0:
                type_nones.append(
                    {
                        'length': int(lines[index - 1].strip('\n').split(' ')[0]) - 1,
                        'mean': numpy.mean(type_none, axis=0),
                        'std': numpy.std(type_none, axis=0),
                        'max': numpy.max(type_none, axis=0),
                        'min': numpy.min(type_none, axis=0)
                    }
                )
            if type_inc.size > 0:
                type_incs.append(
                    {
                        'length': int(lines[index - 1].strip('\n').split(' ')[0]) - 1,
                        'mean': numpy.mean(type_inc, axis=0),
                        'std': numpy.std(type_inc, axis=0),
                        'max': numpy.max(type_inc, axis=0),
                        'min': numpy.min(type_inc, axis=0)
                    }
                )
            if type_dec.size > 0:
                type_decs.append(
                    {
                        'length': int(lines[index - 1].strip('\n').split(' ')[0]) - 1,
                        'mean': numpy.mean(type_dec, axis=0),
                        'std': numpy.std(type_dec, axis=0),
                        'max': numpy.max(type_dec, axis=0),
                        'min': numpy.min(type_dec, axis=0)
                    }
                )

            type_none = []
            type_inc = []
            type_dec = []

        type_none.append(normalize_to_ns(split_line(lines[index])))
        type_inc.append(normalize_to_ns(split_line(lines[index + 1])))
        type_dec.append(normalize_to_ns(split_line(lines[index + 2])))

    type_none = numpy.array(type_none)
    type_inc = numpy.array(type_inc)
    type_dec = numpy.array(type_dec)

    type_nones.append(
        {
            'length': len(type_nones) + 4,
            'mean': numpy.mean(type_none, axis=0),
            'std': numpy.std(type_none, axis=0),
            'max': numpy.max(type_none, axis=0),
            'min': numpy.min(type_none, axis=0)
        }
    )
    type_incs.append(
        {
            'length': len(type_incs) + 4,
            'mean': numpy.mean(type_inc, axis=0),
            'std': numpy.std(type_inc, axis=0),
            'max': numpy.max(type_inc, axis=0),
            'min': numpy.min(type_inc, axis=0)
        }
    )
    type_decs.append(
        {
            'length': len(type_decs) + 4,
            'mean': numpy.mean(type_dec, axis=0),
            'std': numpy.std(type_dec, axis=0),
            'max': numpy.max(type_dec, axis=0),
            'min': numpy.min(type_dec, axis=0)
        }
    )

    draw_error_plot(type_nones, "Bez sortowania")
    draw_error_plot(type_incs, "Sortowanie rosnąco")
    draw_error_plot(type_decs, "Sortowanie malejąco")

    draw_plots_with_labels([
        {
            'label': 'Bez sortowania',
            'list': type_nones
        },
        {
            'label': 'Sortowanie rosnąco',
            'list': type_incs
        },
        {
            'label': 'Sortowanie malejąco',
            'list': type_decs
        },
    ], "Porównanie sortowań")
