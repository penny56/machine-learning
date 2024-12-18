import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 创建Naive Bayes分类器
model = GaussianNB()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 打印分类报告和准确率
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
