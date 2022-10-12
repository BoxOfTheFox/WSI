import json
import numpy

import matplotlib.pyplot as plt

with open('json_data2.json') as json_file:
    data = json.load(json_file)['scores']

for i in range(5, 11):
    scores = numpy.empty((10, 10))
    std = numpy.empty((10, 10))

    for dic in filter(lambda dic: dic['epsilon'] == i / 10, data):
        scores[int(dic['beta'] * 10) - 1][int(dic['gamma'] * 10) - 1] = dic['success rate']
        try:
            std[int(dic['beta'] * 10) - 1][int(dic['gamma'] * 10) - 1] = dic['success rate std'] / dic['success rate']
        except ZeroDivisionError:
            std[int(dic['beta'] * 10) - 1][int(dic['gamma'] * 10) - 1] = 0

    x = numpy.linspace(1, 10, 10) / 10
    y = numpy.linspace(1, 10, 10) / 10
    x, y = numpy.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Success Rate for epsilon: {i/10}")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Success Rate")
    ax.set_zlim(0, 1.1)
    ax.plot_wireframe(x, y, scores, color='orange', label="Success rate")
    ax.plot_wireframe(x, y, std, linestyle='dashed', linewidth=0.5, label="co-efficient of variation")
    ax.legend(loc="best")
    # plt.show()
    plt.savefig(f'podstawa{i}.png')
