from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np

import pandas as pd

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

C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0),
    'KNN classifier': KNeighborsClassifier(C),
    'SVC': SVC(),
    'RFST': RandomForestClassifier(n_estimators=100),
    'ADA': AdaBoostClassifier(n_estimators=100)
    
}

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))