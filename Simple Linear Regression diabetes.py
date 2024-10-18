# From scikit learn web site
# Code source: Jaques Grobler
# License: BSD 3 clause

'''
The example below uses only the first feature of the diabetes（糖尿病） dataset, in order to illustrate the data points within the two-dimensional plot. The straight line can be seen in the plot, showing how linear regression attempts to draw a straight line that will best minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.

The coefficients, residual sum of squares and the coefficient of determination are also calculated.
'''

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#1. decide on the question: 使用的是 scikit-learn 的糖尿病数据集（diabetes dataset），展示了如何使用线性回归进行建模和预测，并计算模型的评估指标。

#2. Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes_X.shape) # 442 行，每行有 10个特征值（10 列）
print(diabetes_X[0])

# Use only one feature (第3列)
diabetes_X = diabetes_X[:, np.newaxis, 2]
print(diabetes_X.shape) # 442 行，每行有 1个特征值（1 列）
print(diabetes_X[0])

# b. Split the data set
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 上面 4行也可以写成一行：
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


#3, 4. Create linear regression object and train
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)


#7. Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

#5. The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

#8. Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()