# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.arange(0, 6, 0.1) # Generate data from 0 to 6 in units of 0.1
y = np.sin(x)

# Plot graph
plt.plot(x, y)
# plt.plot(y, x)
plt.show()