######################### import required linraries

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  # 图像直接显示在 Notebook 的输出单元中，而不需要使用 plt.show()

###################################### Logarithmic 对数 Y = log(X)

X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()          # 对数曲线