#######################################引入包

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score

#1. decide on the question: predict the co2 emissions by a linear regression model

#2. Collect and prepare data
df = pd.read_csv("FuelConsumptionCo2.csv")         # 返回一个dataFrame
print(df.head())                       # 返回这个dataFrame的前5行
print(df.describe())           # 给出这个dataFrame的一些基本属性，如样本数，每种属性的均值之类的

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]  # 选择dataFrame中的4列属性
print(cdf.head(9))             # 返回这4列的前9行

#8. Visualization可视化：直方图
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']] # 只是打乱这4列的顺序
viz.hist()      # 生成直方图，其中 x 轴代表特征值，y 轴代表样本数量

#8. Visualization可视化：散点图
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='green')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#b. 划分数据集
msk = np.random.rand(len(df)) < 0.8     # 80% 的训练集和 20% 的测试集
train = cdf[msk]
test = cdf[~msk]

#3. Choose a training method and train
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

# 输出线性回归模型的 2 个参数：斜率+截距。简单线性模型 y = ax + b
print ('Coefficients: ', regr.coef_)        # 得出：Coefficients:  [[39.90533222]]
print ('Intercept: ',regr.intercept_)       # 得出：Intercept:  [123.3636721]

#8. Visualization可视化：散点图
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='grey')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')      # 点阵图上加一条线，fit line
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#5. 评估 Evaluate :      用test set去评估 training set建模出来的
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

# 输出：
'''
Mean absolute error: 23.20
Residual sum of squares (MSE): 960.17
R2-score: 0.64
'''