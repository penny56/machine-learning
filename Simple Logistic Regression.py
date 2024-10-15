######################### import required linraries

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  # 图像直接显示在 Notebook 的输出单元中，而不需要使用 plt.show()

X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()