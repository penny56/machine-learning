import numpy as np
import matplotlib.pyplot as plt

# non-liner Regression是在Xi和Y之间建立一个非线性关系。通常用k维多项式（X的最大次幂）表示
# 我们画一个3次方的图

x = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3               # 这里，x平方，x立方 就是 Indepdendent Variables （可以看作2个）
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()