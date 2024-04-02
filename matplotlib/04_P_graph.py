# graph of P = 1 - (1 - s^r)^b
import matplotlib.pyplot as plt
import numpy as np

b=5
r=3
s = np.linspace(0,1)
f1 = 1-(1-s**r)**b

b=10
f2 = 1-(1-s**r)**b

b=15
f3 = 1-(1-s**r)**b


plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.plot(s, f1, s, f2, s, f3)
plt.show()