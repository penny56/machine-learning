######################### import required linraries

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  # 图像直接显示在 Notebook 的输出单元中，而不需要使用 plt.show()

# linear Regression 比较简单，在 X 与 Y 之间建立一个线性关系，用一个简单的一维方程式，如 2*(x)+3
x = np.arange(-5.0, 5.0, 0.1) # x = [-5.0, -4.9, -4.8, ..., 4.9]
# You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.figure(figsize=(8,6)) # 创建一个新的绘图窗口，并设置绘图的大小为 8x6 英寸
plt.plot(x, ydata,  'bo') # 蓝色的散点图表示带有噪声的真实数据
plt.plot(x,y, 'r') # 红色的线表示线性模型的拟合线
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()          # 画一个简单的点阵图和一个fit linear
