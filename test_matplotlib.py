
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print("Matplotlib:", matplotlib.__version__)

x = np.arange(10)
y = x ** 2

plt.plot(x, y)
plt.title("Matplotlib Test")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

print("Matplotlib is working successfully!")
