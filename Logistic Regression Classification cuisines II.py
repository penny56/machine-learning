import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from sklearn.svm import SVC
import numpy as np


#1. Decide on the question：根据提供的特征值 X 来预测菜系 y，比如根据：杏仁、当归、茴香...，预测这道 cuisine 是中国菜。

#2. prepare data：将数据分为训练模型所需的 X（译者注：代表特征数据）和 y（译者注：代表标签数据，有没有 y 也是区分 supervised 和 unsupervised 的依据）两个 dataframe。
cuisines_df = pd.read_csv("cleaned_cuisines.csv")
print(cuisines_df.head())

#a. 将 cuisine 列的数据单独保存为的一个 dataframe 作为标签（label）y
cuisines_label_df = cuisines_df['cuisine']
print(cuisines_label_df.head())

#b. 调用 drop() 方法将 Unnamed: 0 和 cuisine 列删除，并将余下的数据作为可以用于训练的特证（feature）X 数据:
cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
print(cuisines_feature_df.head())

#c. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)

#3，4. 选择模型和算法，训练模型
# 这点网站上解释了很多，可以参考一个cheat sheet，包括一些参数。这里我们选择logistic regression
lr = LogisticRegression(multi_class='ovr',solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

#5. 评估模型
accuracy = model.score(X_test, y_test)
print("Accuracy is {}".format(accuracy))

# 也可以通过查看某一行数据（比如第 50 行）来观测到模型运行的情况:
print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
print(f'cuisine: {y_test.iloc[50]}')

# 检查一下这第50行的预测结果的准确率:
test= X_test.iloc[50].values.reshape(-1, 1).T
proba = model.predict_proba(test)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)

topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
print(topPrediction.head())

'''
运行后的输出如下———可以发现这是一道印度菜的可能性最大，是最合理的猜测:

0
indian	0.715851
chinese	0.229475
japanese	0.029763
korean	0.017277
thai	0.007634
'''

# 输出分类报告
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))