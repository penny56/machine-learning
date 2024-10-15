import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
# %matplotlib inline

# 使用pandas dataframe来读数据
my_data = pd.read_csv("drug200.csv", delimiter=",")
print (my_data[0:5])        # 显示5行，有sex, BP, cholesterol，还有Y列：drug X Y C ... 非交互模式下需要 print()

# 1. pre-processing
# a. 创建特征值 feature（X）
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print (X[0:5])

# b. 把sex(性别), BP（血压），Cholesterol(胆固醇)，从字母改为数字：numerical，
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

print (X[0:5])

# c. 创建目标 target（y）
y = my_data["Drug"]

print (y[0:5])

# 2. 划分训练集 + 测试集
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# 3. 选择并训练机器学习模型
# 根据问题选择一个模型，如：决策树、逻辑回归，等，这里使用 DecisionTreeClassifier()
# criterion="entropy": 决策树的分裂标准是信息增益（基于熵的计算），用于选择最佳分裂点。
# max_depth=4：设置决策树的最大深度为 4，限制了树的深度，以防止过拟合。

# a. 初始化模型
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

# b. 训练模型
# Train (fit) the model using the training data
drugTree.fit(X_trainset,y_trainset)

# c. 评估模型

# c1. 使用 model.score(X_test, y_test)进行评估，这个方法是大多数模型自带的评估方法
#     - 对于回归问题，通常返回Rº(决定系数)
#     - 对于分类问题，通常返回准确率
accuracy1 = drugTree.score(X_testset, y_testset)
print(f"模型准确率: {accuracy1:.2f}")

# 3. prediction(预测)

# a 用testing dataset来做预测，并存入叫predTree的变量
predTree = drugTree.predict(X_testset)

# 对比：预测出的 predTree 与测试集 y_testset
print (predTree [0:5])
print (y_testset [0:5])

# 4. Evaluation 评估

# c2. 使用 from sklearn.metrics import accuracy_score 进行预测。这个例子可以一眼对比出 3. 的 predTree v.s. y_testset 的不同。但是多数情况下，需要用这个方法评估 
accuracy2 = metrics.accuracy_score(y_testset, predTree)
print("DecisionTrees's Accuracy: ", accuracy2)

# 5. Visualization 画图

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
# plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
print ()