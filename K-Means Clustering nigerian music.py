import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics


df = pd.read_csv("nigerian-songs.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# top 指的是 df['artist_top_genre'] 列中，每个artist_top_genre（流派）出现的次数，并从高到低排序。
# top的类型是 Series，类似字典
''' top:
artist_top_genre
afro dancehall    206
afropop            61
nigerian pop       19
Name: count, dtype: int64
'''
top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top[:5].index,y=top[:5].values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')
plt.show(block = True)

'''
对你的数据进行聚类
'''

#1. 通过过滤掉丢失的数据来清理，artist_top_genre（流派） 的value 不能是 missing
df = df[df['artist_top_genre'] != 'Missing']
top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top.index,y=top.values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')
plt.show(block = True)

#2. 到目前为止，前三大流派主导了这个数据集。让我们 **专注于** artist_top_genre（流派）== afro dancehall，afropop 和 nigerian pop，另外过滤数据集以删除任何具有 0 流行度值的内容（这意味着它在数据集中没有被归类为流行度并且可以被视为我们的目的的噪音）：
df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
df = df[(df['popularity'] > 0)]
top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top.index,y=top.values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')
plt.show(block = True)

#3. 对数据进行编码，并生成一个相关性热力图：热力图显示数据集中不同特征之间的相关性。颜色越浅表示相关性越强（接近 1 或 -1），颜色越深表示相关性弱（接近 0）。
# 可以看到，唯一强相关性是 energy 和之间 loudness，这并不奇怪，因为嘈杂的音乐通常非常有活力。否则，相关性相对较弱。看看聚类算法可以如何处理这些数据会很有趣。
df.iloc[:, 0:-1] = df.iloc[:, 0:-1].apply(LabelEncoder().fit_transform)
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show(block = True)

'''
数据分布
'''

#8. 使用 Seaborn 的 jointplot() 函数绘制了 popularity（流行度）和 danceability（可舞性）之间的联合分布图，并根据 artist_top_genre（艺术家主要流派）进行着色。因为之前把 df filter 到只有 3 种，所以这个图里只有 3 种颜色的线。
sns.set_theme(style="ticks")

g = sns.jointplot(
    data=df,
    x="popularity", y="danceability", hue="artist_top_genre",
    kind="kde",
)
plt.show(block = True)

#8. 使用 Seaborn 的 FacetGrid 来创建一个带有颜色标记的散点图，其中 popularity（流行度）和 danceability（可舞性）作为两个变量，并通过不同颜色来区分 artist_top_genre（艺术家主要流派）
sns.FacetGrid(df, hue="artist_top_genre", height=3, aspect=3) \
   .map(plt.scatter, "popularity", "danceability") \
   .add_legend()
plt.show(block = True)

'''
K-Means
'''
#1. 准备，看看歌曲数据
#a. 创建一个箱线图，boxplot() 为每一列调用：箱线图（Boxplot） 显示数据分布的五个主要点：最小值、第一四分位数、中位数、第三四分位数和最大值，还能够帮助你快速发现异常值（outliers）。
plt.figure(figsize=(5,5), dpi=200)

plt.subplot(4,3,1)
sns.boxplot(x = 'popularity', data = df)

plt.subplot(4,3,2)
sns.boxplot(x = 'acousticness', data = df)

plt.subplot(4,3,3)
sns.boxplot(x = 'energy', data = df)

plt.subplot(4,3,4)
sns.boxplot(x = 'instrumentalness', data = df)

plt.subplot(4,3,5)
sns.boxplot(x = 'liveness', data = df)

plt.subplot(4,3,6)
sns.boxplot(x = 'loudness', data = df)

plt.subplot(4,3,7)
sns.boxplot(x = 'speechiness', data = df)

plt.subplot(4,3,8)
sns.boxplot(x = 'tempo', data = df)

plt.subplot(4,3,9)
sns.boxplot(x = 'time_signature', data = df)

plt.subplot(4,3,10)
sns.boxplot(x = 'danceability', data = df)

plt.subplot(4,3,11)
sns.boxplot(x = 'length', data = df)

plt.subplot(4,3,12)
sns.boxplot(x = 'release_date', data = df)
plt.show(block = True)
# 您可以浏览数据集并删除这些异常值，但这会使数据非常少。

# 使用 LabelEncoder 对数据进行编码，具体是将分类变量转换为数值型变量，以便后续的数据分析和建模。
le = LabelEncoder()
X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
y = df['artist_top_genre']
X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
y = le.transform(y)

#3,4 选择 KMeans 模型，并训练
# 我们从数据集中挖掘出 3 种歌曲流派，所以让我们尝试 3 种：
nclusters = 3 
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X)

#7. Predict the cluster for each data point
y_cluster_kmeans = km.predict(X)
print(y_cluster_kmeans)

#5. 评估模型
# 看到打印出的数组，其中包含数据帧每一行的预测聚类（0、1 或 2）。使用此数组计算“轮廓分数”：
score = metrics.silhouette_score(X, y_cluster_kmeans)
'''
轮廓系数的值范围从 -1 到 1。
接近 1：表示样本的聚类效果很好，样本在其聚类内比较靠近，且远离其他聚类。
接近 0：表示样本位于两个聚类之间，聚类重叠严重。
接近 -1：表示样本被错误地聚类到其他聚类中。
'''
# 0.2372858587578471
print(score)

#3. 建立模型
# 计算不同聚类数量下的“聚类内平方和”（Within-Cluster Sum of Squares，WCSS），以便使用肘部法则（Elbow Method）确定最佳聚类数量。
'''
range：这些是聚类过程的迭代
random_state：“确定质心初始化的随机数生成。” 
WCSS：“聚类内平方和”测量聚类内所有点到聚类质心的平方平均距离。
Inertia：K-Means 算法尝试选择质心以最小化“惯性”，“惯性是衡量内部相干程度的一种方法”。来源。该值在每次迭代时附加到 wcss 变量。
k-means++：在 Scikit-learn 中，您可以使用“k-means++”优化，它“将质心初始化为（通常）彼此远离，导致可能比随机初始化更好的结果。
'''
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 手肘方法
# 使用手肘方法来确认。
plt.figure(figsize=(10,5))
# 将 range 转换为列表或 NumPy 数组
x_values = list(range(1, 11))
sns.lineplot(x=x_values, y=wcss,marker='o',color='red')
plt.title('Elbow')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# 使用 wcss 您在上一步中构建的变量创建一个图表，显示肘部“弯曲”的位置，这表示最佳聚类数。也许是 3！

# 显示聚类
#a. 再次尝试该过程，这次设置三个聚类，并将聚类显示为散点图：
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
labels = kmeans.predict(X)
plt.scatter(df['popularity'],df['danceability'],c = labels)
plt.xlabel('popularity')
plt.ylabel('danceability')
plt.show()

#b. 检查模型的准确性：
# Result: 95 out of 286 samples were correctly labeled.
# Accuracy score: 0.33
# 准确性差，是因为这些数据太不平衡，相关性太低，列值之间的差异太大，无法很好地聚类。
# 事实上，形成的聚类可能受到我们上面定义的三个类型类别的严重影响或扭曲。那是一个学习的过程！
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))