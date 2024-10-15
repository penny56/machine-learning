import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

#1. Decide on the question：从一个菜系数据集中提取、清理、平衡数据，并生成 cleaned_cuisines.csv 文件

#2. prepare data
df  = pd.read_csv('cuisines.csv')
print(df.head())
print(df.info())

#a. 统计 cuisine（菜系） 列，按出现的 value 的次数（chinese、indian、 japanese...）画柱状图
df.cuisine.value_counts().plot.barh()
plt.show(block=True)

#b. 根据 df 中的cuisine（菜系） 列，将不同cuisine（菜系） 的数据分割成多个 子df ，并输出行列数。
thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')

'''
创建一个食材的数据（ingredient）帧。这个函数会去掉数据中无用的列并按食材的数量(value列)进行分类。

'''
def create_ingredient_df(df):
    #a. 去掉无用列，包括 'cuisine'，并对所有所有食材列的值求和，存入 value 列
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    #b. 去除未使用的食材列，即所有列值为 0 的食材
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    #c. 按照‘value’列（即食材出现次数）降序排序
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False, inplace=False)
    return ingredient_df

#c. 得到理想的每道菜肴最重要的 10 种食材

#c1. for thai
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
plt.show(block=True)

#c2. for japanese
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
plt.show(block=True)

#c3. for chinese
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
plt.show(block=True)

#c4. for indian
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
plt.show(block=True)

#c5. for korean
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
plt.show(block=True)

#d. 去除在不同的菜肴间最普遍的容易造成混乱的食材，大家都喜欢米饭、大蒜和生姜，就去掉。？？？why ？？？
feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine
print(feature_df.head())

#e. 平衡数据集
'''
数据集的平衡指的是确保各个类别的数据数量尽可能相等。当你处理的分类问题中，某些类别的数据量远远多于其他类别时，需要考虑对数据进行平衡。
- 例如，在菜系分类的任务中，某些菜系（如中国菜、印度菜）可能有更多的样本，而一些其他的菜系（如泰国菜、韩国菜）的样本相对较少。如果不进行数据集平衡，模型可能会更倾向于预测数据多的类别，而对数据少的类别表现较差。
- SMOTE 是一种平衡数据集的方法。
'''
#e1. 调用函数 fit_resample(), 此方法通过插入数据来生成新的样本。通过对数据集的平衡，当你对数据进行分类时能够得到更好的结果。现在考虑一个二元分类的问题，如果你的数据集中的大部分数据都属于其中一个类别，那么机器学习的模型就会因为在那个类别的数据更多而判断那个类别更为常见。平衡数据能够去除不公平的数据点。
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)

#e2. 查看每个食材的标签数量:
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')

'''
输出应该是这样的 :
new label count: korean      799
chinese     799
indian      799
japanese    799
thai        799
Name: cuisine, dtype: int64
old label count: korean      799
indian      598
chinese     442
japanese    320
thai        289
Name: cuisine, dtype: int64
现在这个数据集不仅干净、平衡而且还很“美味” !
'''

#e3. 保存你处理过后的平衡的数据（包括标签和特征），将其保存为一个可以被输出到文件中的数据帧:
transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')

#e4. 你可以通过调用函数 transformed_df.head() 和 transformed_df.info() 再检查一下你的数据。 接下来要将数据保存以供在未来的课程中使用:
transformed_df.head()
transformed_df.info()
transformed_df.to_csv("cleaned_cuisines.csv")