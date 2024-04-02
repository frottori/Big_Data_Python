# importing the required modules
import matplotlib.pyplot as plt
import numpy as np

# setting the x - coordinates
x = np.arange(0, 2*(np.pi), 0.1)
# setting the corresponding y - coordinates
y = np.sin(x)

plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Sine graph')

plt.plot(x, y)
plt.show()