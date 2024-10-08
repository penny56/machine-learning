import numpy as np
import matplotlib.pyplot as plt

# 1. Generating Some Sample Data
# a. Generate 100 samples
n_samples = 100

# b. Generate 100 random input data (X) - 100 samples (rows),
#    1 indicates the number of columns in the array.
#    it will create a 2D array with 100 rows and 1 column.
#    X 应该是这个样子：
'''
array([[1.23456789],
       [0.98765432],
       ...
       [0.3456789 ]])
'''
X = 2 * np.random.rand(n_samples, 1)

# a. Generate corresponding target values (y) with some noise
# 使用线性关系 y=4+3Xy=4+3X 生成对应的目标值 yy，并加上了正态分布的噪声，以模拟现实中的数据不完美性。
# y 应该是这个样子：
'''
array([[ 6.23456789],
       [ 5.98765432],
        ...
       [ 4.3456789 ]])
'''
y = 4 + 3 * X + np.random.randn(n_samples, 1)



# 2. Implementing Linear Regression Using the Normal Equation
# a. 为 X 增加了一列全为1的偏置项(bias term），第一列为 X的原始值，第二列全为 1
# X_b 应该是这个样子：
'''
array([[1.        , 1.24896252],
       [1.        , 0.52691165],
       ...
       [1.        , 0.53547772]])
'''
X_b = np.c_[np.ones((n_samples, 1)), X]  # shape becomes (100, 2)

# b. Calculate the optimal parameters using the Normal Equation
# 通过正规方程计算最优参数 theta_best，包括截距和斜率。不同的 X y 组合，会算出不同的theta_best。
# 在计算theta_best之前，为什么要将 X 变成 X_b？是因为要引入偏置项(bias term），将模型表示得更加简洁，也方便了后续的计算。
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Display the resulting parameters (theta)
# theta_best 应该是这个样子：
'''
Optimal parameters (theta):
[[4.03845804]
 [2.94374602]]
'''
print(f"Optimal parameters (theta):\n{theta_best}")

# 3. Making Predictions**
# 生成新的数据，用上一 step 生成的 theta_best 来做预测试试
# a. 生成新数据
X_new = np.array([[0], [2]])
# b. 增加偏置项(bias term）
X_new_b = np.c_[np.ones((2, 1)), X_new]

# c. 利用 theta_best 来预测 target values
y_predict = X_new_b.dot(theta_best)

# y_predict 应该是这个样子：
# 注意，这里的 X_new 分别是 [0, 2]，经过linear 公式算出 y = 4 + 3 * X 得出 y 应该是 [4, 10]，可以看到与 y_predict 很接近。
'''
Predicted values:
[[4.33900236]
 [9.59933983]]
'''
print(f"Predicted values:\n{y_predict}")

# 4. Plotting the Results 画结果
# Plot the original data
plt.plot(X, y, "b.")

# Plot the regression line
plt.plot(X_new, y_predict, "r-", linewidth=2)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression with NumPy")
plt.show()
