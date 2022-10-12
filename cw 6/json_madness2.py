import json

import numpy as np
from matplotlib import pyplot as plt

with open('json_data2.json') as json_file:
    data1 = json.load(json_file)['scores']

with open('json_data3.json') as json_file:
    data2 = json.load(json_file)['scores']

empty = {'beta': 0.0, 'gamma': 0.0, 'epsilon': 0.0, 'success rate': 0.0, 'success rate std': 0.0}
dic1 = [empty, empty, empty, empty]

for i in range(1, 11):
    for dic in filter(lambda dic: dic['beta'] == i / 10 and dic['gamma'] == i / 10 and dic['epsilon'] == i / 10, data1):
        dic1.append(dic)

# Numbers of pairs of bars you want
N = len(data2)

# Data on X-axis

# Specify the values of blue bars (height)
blue_bar = list(map(lambda dic: dic['success rate'], dic1))
# Specify the values of orange bars (height)
orange_bar = list(map(lambda dic: dic['success rate'], data2))

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10, 5))

# Width of a bar
width = 0.3

# Plotting
plt.bar(ind, blue_bar, width, label='Podstawowe nagrody')
plt.bar(ind + width, orange_bar, width, label='Customowe nagrody')

plt.xlabel('Wartości epsilon, gamma, beta')
plt.ylabel('Success rate')
plt.title('Porównanie wpływu nagród na jakość modelu')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
# plt.xticks(ind + width / 2, ('Xtick1', 'Xtick3', 'Xtick3'))
plt.xticks(ind + width / 2, [k / 10 for k in range(1, 11)])

# Finding the best position for legends and putting it
plt.legend(loc='best')
# plt.show()
plt.savefig('porownanie.png')
