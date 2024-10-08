import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 2. 拆分数据集为训练集和测试集，30%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3. 创建k-NN分类器
# a. 初始化
k = 3  # 选择k值为3，就是邻居数量为 3
model = KNeighborsClassifier(n_neighbors=k)

# b. 训练模型
model.fit(X_train, y_train)

accuracy4D = model.score(X_test, y_test)
print(f"模型准确率4D: {accuracy4D:.2f}")

# 4. 进行预测
y_pred = model.predict(X_test)

# 5. 打印分类报告和准确率
print("分类报告4D:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(f"准确率4D: {accuracy_score(y_test, y_pred):.2f}")

# 可视化预测结果（二维特征示例），这里只选择 前两个 特征进行可视化
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

# 2.b 重新训练模型只用前两个特征，可以看到模型准确率从 4D 的 0.92 降到 0.73。
model.fit(X_train_2d, y_train)

accuracy2D = model.score(X_test_2d, y_test)
print(f"模型准确率2D: {accuracy2D:.2f}")

# 创建一个二维网格，用来可视化预测结果。xx 和 yy 是二维平面上的坐标点。
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 4. 重新进行预测
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. 重新打印分类报告和准确率(这里出问题了，可以找到为什么吗？)
#print("分类报告2D:")
#print(classification_report(y_test, Z, target_names=iris.target_names))
#print(f"准确率2D: {accuracy_score(y_test, Z):.2f}")

# 6. 绘制决策边界
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.title(f'k-NN 分类器 (k={k}) - 前两个特征的决策边界')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.show()
