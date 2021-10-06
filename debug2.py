import numpy as np

i = np.array([[0, 1, 2], [0, 1, 1]])
x = np.arange(9).reshape(3, 3)
print(x)
print(x[i, i])
