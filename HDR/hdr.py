import numpy as np
import matplotlib.pyplot as plt

with open('./data/responseCurve.txt', 'r') as f:
    response = np.array([float(i) for i in f.read().split('\n') if i != ''], dtype=np.float32)


y = np.arange(response.size)

plt.plot(response, y)
plt.xscale('log')
plt.show()
