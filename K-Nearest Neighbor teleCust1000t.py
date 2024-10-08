import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# %matplotlib inline

# 1. pre-processing
# a. load data from CSV file
df = pd.read_csv('teleCust1000t.csv')
print (df.head())           # 显示dataset前5行，有各种属性 region tenure age marital address income ... ，最后一列是custcat，就是Y，客户类别

# b. Data Visualization and Anylisis 数据可视化和分析
# 看一下每个custcat有多少客户，一共4类custcat，分别是1，2，3，4。
print (df['custcat'].value_counts())  # 输出每个custcat的客户数量: 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

df.hist(column='income', bins=50)   # 画个图，不同income的用户数的柱图

# c. Feature dataset, 定义feature set X; target y
print (df.columns)      # 这个set包含dataset中的那些列

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print (X[0:5])

y = df['custcat'].values
print (y[0:5])          # 显示custcat的array

# d. 对特征数据 X 进行标准化处理 Normalize Data，使得：
#    - 每个特征列的均值为 0。
#    - 每个特征列的标准差为 1。
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print (X[0:5])

# 2. 划分训练集 + 测试集
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# 3. 选择并训练机器学习模型
# a+b 初始化+训练
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print (neigh)

# c. 评估模型
accuracy1 = neigh.score(X_train, y_train)
print(f"模型准确率: {accuracy1:.2f}")

# 4. Predicting 进行预测
yhat = neigh.predict(X_test)
print (yhat[0:5])       # 输出 array([1, 1, 3, 2, 4])
print (y_test[0:5])

# 5. Accuracy evaluation
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

'''
输出：
Train set Accuracy:  0.5475
Test set Accuracy:  0.32
'''