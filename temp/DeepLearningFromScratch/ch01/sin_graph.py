# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# Plot graph
plt.plot(x, y)
plt.show()
