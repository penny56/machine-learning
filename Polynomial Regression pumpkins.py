import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#1. 定义问题（Decide on the question）
# 预测给定月份内待售南瓜的价格

#2. 收集和准备数据（Collect and prepare data）
pumpkins = pd.read_csv('US-pumpkins.csv')
print(pumpkins.head())

# 查找每一列丢失数据的 count，可能是无关紧要
print(pumpkins.isnull().sum())

# 清洗数据，只保留一些重要的列，删除无关紧要的列
new_columns = ['Package', 'Month', 'Low Price', 'High Price', 'Date']
pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)
print(pumpkins.head())

# 确定平均价格，Data 列转换成只显示月份
price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
month = pd.DatetimeIndex(pumpkins['Date']).month
print(month)

# 将转换后的数据复制到新的 df 中
new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
print(new_pumpkins)

# 可以看到，package（包装）这一列的单位不统一。打印一下用'bushel'作单位的数量
print(pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)])

# 只有 415 行，而总数有 1758 行。
# 如果 package 包含'1 1/9'，价格就除以(1 + 1/9)；如果包含'1/2'，价格就除以(1/2)
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)

#8. 可视化
#a. 散点图（无意义）
price = new_pumpkins.Price
month = new_pumpkins.Month
plt.scatter(price, month)
plt.show()

#b. 柱状图（意义不大）
new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
plt.ylabel("Pumpkin Price")
plt.show()

#2. 字符串现在都是数字。这让你更难阅读，但对 Scikit-learn 来说更容易理解！
new_pumpkins.iloc[:, 0:-1] = new_pumpkins.iloc[:, 0:-1].apply(LabelEncoder().fit_transform)
print(new_pumpkins['Package'].corr(new_pumpkins['Price']))

new_pumpkins.dropna(inplace=True)
new_pumpkins.info()

# for linear regression: 创建最小全集并打印
new_columns = ['Package', 'Price']
lin_pumpkins = new_pumpkins.drop([c for c in new_pumpkins.columns if c not in new_columns], axis='columns')
print(lin_pumpkins)

# for linear regression: 分配 X 和 y 的坐标数据
X = lin_pumpkins.values[:, :1]
y = lin_pumpkins.values[:, 1:2]

# 3,4,5 for linear regression: 划分数据集，选择模型，训练模型，并计算accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

pred = lin_reg.predict(X_test)

accuracy_score = lin_reg.score(X_train,y_train)
print('Model Accuracy: ', accuracy_score) # linear regression: 0.15670291028939276 线性的 model 太差了

#8. for linear regression: 画出来
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, pred, color='blue', linewidth=3)

plt.xlabel('Package')
plt.ylabel('Price')

plt.show()

#7. for linear regression: predict for linear regression
print(lin_reg.predict( np.array([ [2.75] ]) ))

#2. for polynomial regression 
#new_columns = ['Variety', 'Package', 'City', 'Month', 'Price']
new_columns = ['Package', 'Price']
poly_pumpkins = new_pumpkins.drop([c for c in new_pumpkins.columns if c not in new_columns], axis='columns')

print(poly_pumpkins)

#8. visualization 可视化。可是没有显示出来东西。
corr = poly_pumpkins.corr()
corr.style.background_gradient(cmap='coolwarm')

# 创建管道：它是一个估计器链。在这种情况下，管道包括多项式特征或形成非线性路径的预测。
#a. 构建 X 和 y 列：
X=poly_pumpkins.iloc[:,0:1].values
y=poly_pumpkins.iloc[:,1:2].values

#b. 通过调用 make_pipeline() 方法创建管道：
pipeline = make_pipeline(PolynomialFeatures(4), LinearRegression())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipeline.fit(np.array(X_train), y_train)

y_pred=pipeline.predict(X_test)

# 创建序列：此时，你需要使用_排序好的_数据创建一个新的 dataframe ，以便管道可以创建序列。
df = pd.DataFrame({'x': X_test[:,0], 'y': y_pred[:,0]})
df.sort_values(by='x',inplace = True)
points = pd.DataFrame(df).to_numpy()

plt.plot(points[:, 0], points[:, 1],color="blue", linewidth=3)
plt.xlabel('Package')
plt.ylabel('Price')
plt.scatter(X,y, color="black")
plt.show()

#5. Accuracy 模型评估
accuracy_score = pipeline.score(X_train,y_train)
print('Model Accuracy: ', accuracy_score) # ploynomial regression = 0.4901761778380248 比 linear 高多了。

#7. predict 
print(pipeline.predict( np.array([ [2.75] ]) ))
