import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 创建示例数据

np.random.seed(0) # 设置随机种子以确保结果可重现
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10],
    [9, 10, 11],
    [10, 11, 12]
])
y = np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5])

# 拆分数据集，分为训练集和测试集。test_size=0.2表示测试集占总数的20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 打印模型预测结果
print("测试集真实值和预测值:")
for real, pred in zip(y_test, y_pred):
    print(f"真实值: {real:.2f}, 预测值: {pred:.2f}")

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印评估结果
print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")
# 可视化预测结果
plt.figure(figsize=(10, 6))

# 绘制真实值和预测值的散点图
plt.scatter(range(len(y_test)), y_test, color='blue', label='真实值', marker='o')
plt.scatter(range(len(y_test)), y_pred, color='red', label='预测值', marker='x')

# 添加标题和标签
plt.title('real v.s. predict')
plt.xlabel('sample index')
plt.ylabel('value')
plt.legend()

# 显示图形
plt.show()
