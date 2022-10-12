import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

x = np.linspace(1,10,10)/10

y = np.linspace(1,10,10)/10

x, y = np.meshgrid(x,y)

z = np.random.randn(10,10)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(x, y, z)

plt.show()
