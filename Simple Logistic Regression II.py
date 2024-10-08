import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print (os.getcwd())


df = pd.read_csv("./china_gdp.csv")
print (df.head(10))         # 输出前10行，每行2列，Year | Value(GDP)。如：1965	6.970915e+10。在非交互环境里，需要 print 来显示。


###################################### Plotting the Dataset 画dataset
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()  # 画一个点阵图，可以看出来，2005年之后发展快，2010年之后发展变慢

###################################### choose a model 选择一个模型

# 决定选择logistic (逻辑模型)模型，因为它具有从缓慢增长开始，在中间增加增长，然后在结束时再次减少的特性

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

####################################### Building The Model 创建模型

# 创建模型，初始化参数
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

# Lets look at a sample sigmoid line that might fit with the data       看一下sigma型样本
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')              # 输出GDP点阵图和一条logistic fit line，但两者差别很大

# 现在的任务是找出最好的参数，现在规范X 和 Y
# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)



####################################### 如何找为fit line找出最佳参数？？？ ==> 使用 curve_fit

# 用 curve_fit 可以non-linear fit sigmoid function
from scipy.optimize import curve_fit
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


###################################### 练习：计算 model的 accuracy

# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )

'''
输出：
Mean absolute error: 0.03
Residual sum of squares (MSE): 0.00
R2-score: 0.98
'''