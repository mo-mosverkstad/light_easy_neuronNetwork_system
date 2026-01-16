# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.arange(0, 6, 0.1) # Generate data from 0 to 6 in units of 0.1
y1 = np.sin(x)
y2 = np.cos(x)

# Plot graph
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")
plt.xlabel("x") # x-axis label
plt.ylabel("y") # y-axis label
plt.title('sin & cos')
plt.legend()
plt.show()