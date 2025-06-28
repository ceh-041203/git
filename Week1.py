import pandas as pd
# 检查数据
df = pd.read_csv('nigerian-songs.csv')
print('数据基本信息：',df.info())

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