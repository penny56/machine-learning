#######################################引入包

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %matplotlib inline

#######################################读入数据

df = pd.read_csv("FuelConsumptionCo2.csv")         # 返回一个dataFrame

# take a look at the dataset
df.head()                       # 返回这个dataFrame的前5行

#######################################探索数据

# summarize the data
df.describe()           # 给出这个dataFrame的一些基本属性，如样本数，每种属性的均值之类的

# Lets select some features to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]  # 选择dataFrame中的4列属性
cdf.head(9)             # 返回这4列的前9行

# we can plot each of these fearues:
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']] # 打乱这4列的顺序？？？
viz.hist()      # 为4列属性生成4个矩形图。可是x轴y轴是什么看不明白啊。
plt.show()

# Now, lets plot each of these features vs the Emission, to see how linear is their relation:
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()  # x轴：FUELCONSUMPTION_COMB，y轴：CO2EMISSIONS，画点阵图，感觉有3条明显放射线

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()  # x轴：ENGINESIZE，y轴：CO2EMISSIONS，画点阵图，感觉有是一片放射点区域

###################################### 练习

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='green')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()  # x轴：ENGINESIZE，y轴：CO2EMISSIONS，画点阵图，感觉有是一片放射点区域

###################################### 创建train/test dataset

msk = np.random.rand(len(df)) < 0.8     # 应该是80% dataset作为train，20% dataset作为test
train = cdf[msk]
test = cdf[~msk]

###################################### Train data distribution

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')    # 用选好的training set来画点阵图
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

###################################### 建模

# Using sklearn package to model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)        # 得出：Coefficients:  [[39.90533222]]         系数
print ('Intercept: ',regr.intercept_)       # 得出：Intercept:  [123.3636721]              窃听

# 在单一线性回归中，这两个参数就可以画出 fit line。

###################################### 画输出

# we can plot the fit line over the data:
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')      # 点阵图上加一条线，fit line
plt.xlabel("Engine size")
plt.ylabel("Emission")

###################################### 评估       用test set去评估 training set建模出来的

# 有多种评估方式，这里使用MSE

from sklearn.metrics import r2_score

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