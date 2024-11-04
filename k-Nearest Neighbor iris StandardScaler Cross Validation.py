# 黑马程序员
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 特征预处理时，加一个标准化预处理，使之无量纲化
from sklearn.preprocessing import StandardScaler

# 加入风格搜索与交叉验证
from sklearn.model_selection import GridSearchCV

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 2. 拆分数据集为训练集和测试集，30%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 2.1 对 x_train, x_test 做标准化，无量纲化
'''
解释一下：对数据集做fit() and transform()，为什么对 x_train是用fit_transform(), 而对 x_test 使用的是 transform()?
分析：
- fit()：在训练集上计算标准化所需的统计信息，例如均值和标准差。
- transform()：使用计算出的均值和标准差对训练集 X_train 进行标准化。
答：对训练集用 fit_transform() 是为了学习数据特征，同时变换数据；对测试集仅用 transform() 是为了确保测试集使用相同的转换规则，保持数据独立性。
'''
transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)

# 3. 创建k-NN分类器
# a. 初始化
k = 3  # 选择k值为3，就是邻居数量为 3

# 因为加入风格搜索与交叉验证，所以这里不需要k值
model = KNeighborsClassifier()

# 6. 加入风格搜索与交叉验证
param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
model = GridSearchCV(model, param_grid=param_dict, cv=10)

# b. 训练模型
model.fit(X_train, y_train)

# 4. 进行预测
y_predict = model.predict(X_test)

# 5. 模型评估
# a. 方法1: 直接比对真实值和预测值
print("y_predict:\n", y_predict)
print("直接对比真实值与预测值：\n", y_test == y_predict)

# b. 方法2: 计算准确率
score = model.score(X_test, y_test)
print("准确率为：\n", score)

# 打印：
print("最佳参数:\n", model.best_params_)
print("最佳结果:\n", model.best_score_)
print("最佳估计器:\n", model.best_estimator_)
print("最佳交叉验证结果:\n", model.cv_results_)

