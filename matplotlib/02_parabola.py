import matplotlib.pyplot as plt
import numpy as np

# Creating vectors X and Y
x = np.linspace(-2, 2, 100)
y = x ** 2
fig = plt.figure(figsize = (10, 5))

# Create the plot
plt.plot(x, y)
# Show the plot
plt.show()