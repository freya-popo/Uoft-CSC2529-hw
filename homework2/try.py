import numpy as np
import math
import matplotlib.pyplot as plt
x = [[0.07511361, 0.1238414, 0.07511361],
     [0.1238414, 0.20417996, 0.1238414],
     [0.07511361, 0.1238414, 0.07511361]]

y = [[0., 0., 0.],
     [0., 0.0745098, 0.24705882],
     [0., 0.05882353, 0.14509804]]
z = 0
F = 0
f = np.zeros_like(x)
for i in range(len(x)):
    for j in range(len(x[0])):
        # print(x[i][j]*y[i][j])
        # z += x[i][j] * y[i][j]
        f[i][j] = math.exp(-(y[i][j] - y[1][1]) ** 2 / (2 * (0.25 ** 2)))
        # print(f[i][j])
        F += y[i][j] * f[i][j] * x[i][j]

        z += x[i][j] * f[i][j]
print(z)
print('F', F)
print(F / z)

fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(50, 40))
plt.show()