import pandas as pd
# 检查数据
df = pd.read_csv('nigerian-songs.csv')
print('数据基本信息：',df.info())
# 数据预处理，选择所有数字特征
X = df.drop(columns=['name', 'album', 'artist', 'artist_top_genre'])