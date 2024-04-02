import matplotlib.pyplot as plt
import numpy as np

# x-axis values
x = [1,2,3,4,5,6,7, 8,9,10]
# y-axis values
y = [2,4,5,7,6,8,9,11,12,12]
# plotting points as a scatter plot
plt.scatter(x, y, label = "stars", color= "green",marker= "*", s=30)


plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('My scatter plot!')
plt.legend()
plt.show()