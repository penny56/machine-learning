import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# two more modules for 评估指标
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 特征：房屋面积（平方英尺），reshape(-1,1)会将一个一维数组转化为一个二维数组(二维一列)。
X = np.array([600, 800, 1000, 1200, 1400]).reshape(-1, 1)

# 目标：房屋价格（万元）
y = np.array([50, 70, 80, 100, 120])



# 下列代码段用于评估一个回归模型训练情况

# 划分数据集为训练集和测试集，random_state = 随机种子，用于确保每次运行都能得到相同的划分结果，保证结果的可重复性。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建和训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 打印评估结果
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")



# 创建线性回归模型
model = LinearRegression()

# 训练模型，X是特征数据，y是目标数据
model.fit(X, y)

# 预测面积为1500平方英尺的房屋价格，通过训练好的module进行预测。
predicted_price = model.predict([[1500]])
print(f"预测的房屋价格: {predicted_price[0]:.2f} 万元")

# 绘制数据点和回归直线
plt.scatter(X, y, color='blue', label='real data')  # 原始数据点
plt.plot(X, model.predict(X), color='red', label='linear regression')  # 回归直线
plt.xlabel("area(inch2)")
plt.ylabel("prise(10K)")
plt.title("area and prise linear regression")
plt.legend()
plt.show()
