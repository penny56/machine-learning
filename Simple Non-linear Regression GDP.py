import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

#1. 这段代码实现了对中国GDP数据的处理和拟合，使用了逻辑回归（Logistic Regression）模型来预测GDP的增长趋势。
# 注意：SciPy 和 scikit-learn（sklearn）都可以用于机器学习，这里使用的是 SciPy

#2. prepare data
df = pd.read_csv("./china_gdp.csv")
print (df.head(10))         # 输出前10行，每行2列，Year | Value(GDP)。如：1965	6.970915e+10。在非交互环境里，需要 print 来显示。

# Plotting the Dataset 画dataset
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()  # 画一个点阵图，可以看出来，2005年之后发展快，2010年之后发展变慢

#3. 选择模型及算法
'''
决定选择logistic (逻辑模型)模型，因为它具有从缓慢增长开始，在中间增加增长，然后在结束时再次减少的特性
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X)) # 这个公式就是模拟 sigmoid() 的输出

这里画一个 logistic 示意图
plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
'''

#a. 创建模型
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y


#b. 初始预测，之后还要通过生成的 Y_pred 与实际数据进行比较，如果初步的预测线与实际数据的拟合效果不佳，表明参数需要优化。
# 根据样本数据，选择 sigma 参数。
beta_1 = 0.10
beta_2 = 1990.0

Y_pred = sigmoid(x_data, beta_1 , beta_2)

# 画图做对比：
plt.plot(x_data, Y_pred*15000000000000.)    # 画出 初始预测 的蓝线
plt.plot(x_data, y_data, 'ro')              # 画出 样本数据 的红点，两者对比！

#6. 模型调优（Parameter tuning）：找出最好的参数，现在规范 X 和 Y
# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# 使用curve_fit进行非线性拟合，找到最佳参数。
# curve_fit通过最小化预测值与实际值之间的差异来优化sigmoid函数的参数，返回最佳参数popt和参数协方差pcov。
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))         # 输出： beta_1 = 690.453017, beta_2 = 0.997207

# 画出 结论
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()                  # 画出GDP点阵和一条fit line，这次二者非常吻合


#2. split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

#3. build the model using train set ？？？ 这里需要重新生成参数吗？那之前的调优不就是用不到了吗？
popt, pcov = curve_fit(sigmoid, train_x, train_y)

#4. predict using test set（使用之前调优过的 popt!）
y_hat = sigmoid(test_x, *popt)

#5. evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , test_y) )

'''
输出：
Mean absolute error: 0.03
Residual sum of squares (MSE): 0.00
R2-score: 0.98
'''