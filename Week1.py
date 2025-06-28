import pandas as pd
# 检查数据
df = pd.read_csv('nigerian-songs.csv')
print('数据基本信息：',df.info())

print('————————————————————————————————————————')

# 数据预处理，选择所有数字特征
X = df.drop(columns=['name', 'album', 'artist', 'artist_top_genre'])

# 分析k-means的聚类效果
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# 使用轮廓系数选择最佳的簇数
silhouette_scores = []
for n_cluster in range(2, 16):
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_scores.append([n_cluster, silhouette_score(X, labels)])

silhouette_scores = pd.DataFrame(silhouette_scores, columns=['簇数', '轮廓系数'])
best_n_cluster = silhouette_scores[silhouette_scores['轮廓系数'] == silhouette_scores['轮廓系数'].max()]['簇数'].values[0]

print('不同簇数对应的轮廓系数：')
print(silhouette_scores)
print(f'最佳簇数: {best_n_cluster}')

print('————————————————————————————————————————')

# 可视化不同簇数对应的轮廓系数
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适配中文
plt.rcParams["axes.unicode_minus"] = False # 确保负号正确显示

plt.figure(figsize=(10, 6))
plt.plot(silhouette_scores['簇数'], silhouette_scores['轮廓系数'], marker='o', color='b', label='轮廓系数')

# 添加数据标签
for x, y in zip(silhouette_scores['簇数'], silhouette_scores['轮廓系数']):
    plt.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0, 5), ha='center')

plt.xlabel('簇数')
# 设置 x 轴刻度显示所有簇数
plt.xticks(silhouette_scores['簇数'], rotation=45)
plt.ylabel('轮廓系数')
plt.title('不同簇数对应的轮廓系数')
plt.grid(True)
plt.legend()
# 突出最佳簇数对应的点
best_index = silhouette_scores[silhouette_scores['簇数'] == best_n_cluster].index[0]
plt.scatter(silhouette_scores['簇数'][best_index], silhouette_scores['轮廓系数'][best_index], color='r', zorder=5, label=f'最佳簇数({best_n_cluster})')
plt.show()

# k-means初始值是否敏感问题探索
import numpy as np
# 多次运行k-means，使用不同的随机初始值
n_runs = 10
inertia_values = []
for _ in range(n_runs):
    kmeans = KMeans(n_clusters=best_n_cluster, init='random')
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

print(f'不同初始值下的惯性值: {inertia_values}')
if np.std(inertia_values) > 100:  # 可以根据实际情况调整阈值
    print('结论：k-means对初始值敏感')
else:
    print('结论：k-means对初始值不敏感')

print('————————————————————————————————————————')

# k-means是否对数据归一敏感问题探索
# 对数据进行归一化
from sklearn.preprocessing import StandardScaler
# 对数据进行归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=best_n_cluster)
kmeans_scaled.fit(X_scaled)
labels_scaled = kmeans_scaled.labels_
silhouette_score_scaled = silhouette_score(X_scaled, labels_scaled)

print(f'归一化前的轮廓系数: {silhouette_scores[silhouette_scores["簇数"] == best_n_cluster]["轮廓系数"].values[0]}')
print(f'归一化后的轮廓系数: {silhouette_score_scaled}')
if abs(silhouette_score_scaled - silhouette_scores[silhouette_scores["簇数"] == best_n_cluster]["轮廓系数"].values[0]) > 0.1:  # 可以根据实际情况调整阈值
    print('结论：k-means对数据归一敏感')
else:
    print('结论：k-means对数据归一不敏感')

print('————————————————————————————————————————')

# 特征相关性分析
# 计算特征之间的相关性
correlation_matrix = X.corr()
# 绘制特征相关性热力图
import seaborn as sns
# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适配中文
# 确保负号正确显示
plt.rcParams["axes.unicode_minus"] = False
# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('特征相关性热力图')
plt.show()

# 关于PCA降维的理解
# 1.使用PCA降维到2维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print('各主成分的方差解释比例: ', pca.explained_variance_ratio_)
print('累积方差解释比例: ', np.cumsum(pca.explained_variance_ratio_))

print('————————————————————————————————————————')

# 2.绘制碎石图确定合适的主成分数量
pca_full = PCA().fit(X_scaled)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         pca_full.explained_variance_ratio_, 'o-')
plt.xlabel('主成分数量')
plt.ylabel('方差解释比例')
plt.title('碎石图')
plt.grid(True)
plt.show()

# 3. 根据需求选择降维数
# 使用PCA降维到4维
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
print('各主成分的方差解释比例: ', pca.explained_variance_ratio_)
print('累积方差解释比例: ', np.cumsum(pca.explained_variance_ratio_))

print('————————————————————————————————————————')

# 4. 以80%的累积方差为解释目标，判断需要多少主成分才能保留足够信息
pca_more = PCA(n_components=min(X_scaled.shape)).fit(X_scaled)
cumulative_variance = np.cumsum(pca_more.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
plt.axhline(y=0.8, color='r', linestyle='--', label='80%阈值')
plt.xlabel('主成分数量')
plt.ylabel('累积方差解释比例')
plt.title('累积方差解释比例图')
plt.legend()
plt.grid(True)
plt.show()

# 5. 根据需求选择降维数
# 使用PCA降维到8维
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)
print('各主成分的方差解释比例: ', pca.explained_variance_ratio_)
print('累积方差解释比例: ', np.cumsum(pca.explained_variance_ratio_))