from matplotlib import pyplot as plt # pip install matplotlib

x_values = [1, 2, 3, 4]
y_values = [5, 4, 6, 2]

plt.plot(x_values, y_values)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Two Lines')
plt.show()