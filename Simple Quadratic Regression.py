######################### import required linraries

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  # 图像直接显示在 Notebook 的输出单元中，而不需要使用 plt.show()

####################################### Quadratic Y = X平方

x = np.arange(-5.0, 5.0, 0.1)

# You can adjust the slope and intercept to verify the changes in the graph
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()          # 反抛物线