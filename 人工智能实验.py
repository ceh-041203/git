##*数据预处理*
import numpy as np
import pandas as pd

##*数据预处理*
df = pd.read_csv('uwide.csv')
#df.info()
#print(df.describe())
#将性别等中文字段进行因子化（Factorize）处理为数字型变量
factor = pd.factorize(df['SEX'])
df.SEX = factor[0]
#查看样本中的空值情况
null_columns = df.columns[df.isnull().any()]
#print(df[df.isnull().any(axis=1)][null_columns].head())
#对所有特征值为空的样本以0填充
df=df.fillna(0)
#生成标签列，以课程的总分作为标准，<60分即为1，>60即为0，将其存入一个列名为SState的字段，作为学习失败与否的标签。
df["SState"]=np.where(df["TOTALSCORE"]>60,0,1)
cols = df.columns.tolist()
#去除无关列
df = df[['BROWSER_COUNT','COURSE_COUNT','COURSE_SUM_VIEW','COURSE_AVG_SCORE',
          'EXAM_AH_SCORE','EXAM_WRITEN_SCORE','EXAM_MIDDLE_SCORE','EXAM_LAB',
          'EXAM_PROGRESS','EXAM_GROUP_SCORE','EXAM_FACE_SCORE','EXAM_ONLINE_SCORE',
          'NODEBB_CHANNEL_COUNT','NODEBB_TOPIC_COUNT','COURSE_SUM_VIDEO_LEN',
          'SEX','EXAM_HOMEWORK','EXAM_LABSCORE','EXAM_OTHERSCORE','NODEBB_PARTICIPATIONRATE',
          'COURSE_WORKTIME','COURSE_WORKACCURACYRATE','COURSE_WORKCOMPLETERATE',
          'NODEBB_POSTSCOUNT','NODEBB_NORMALBBSPOSTSCOUONT','NODEBB_REALBBSARCHIVECOUNT',
          'NORMALBBSARCHIVECOUNT','COURSE_WORKCOUNT','SState']]
#print(df.columns.tolist())
#print(df.SState.value_counts())

##*分类不平衡及标准化处理*
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
'''
#****************
# 二分类平衡
df_majority = df[df.SState==0]
df_minority = df[df.SState==1]
count_times=8
df_majority_downsampled = df_majority
# 下样本多数类
if len(df_majority)>len(df_minority)*count_times:
    new_majority_count = len(df_minority)*count_times
    df_majority_downsampled =resample(df_majority,
                                      replace=False,#相同则替代
                                      n_samples=new_majority_count,# 匹配少数类
                                      random_state=123)
# 将少数类与下采样多数类合并
df = pd.concat([df_majority_downsampled, df_minority])
#*****************
'''
# 选择特征（X）和标签（y）
X = df.iloc[:, :-1].values  # 所有行，除了最后一列的所有列
y = df.iloc[:, -1].values   # 所有行，最后一列
#划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sm = SMOTE(random_state=42)
print('Original dataset shape %s' % Counter(y_train))
X_res, y_res = sm.fit_resample(X_train,y_train)
print('Resampled dataset shape %s' % Counter(y_res))
#新生成的样本覆盖训练集
X_train = X_res
y_train = y_res
#对所有输入变量采用sklearn中的StandardScaler标准化方法转换。
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##*随机森林算法*
from sklearn.metrics import make_scorer,precision_score,recall_score,accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from mlxtend.evaluate import confusion_matrix
import sklearn
#模型采用随机森林算法，并使用网格搜索（grid search）进行优化
param_grid = {
    'min_samples_split': range(2, 10),
    'n_estimators' : [10,50,100,150],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1, 2, 4]
}
# 网格搜索法确定最佳参数
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,verbose=2,n_jobs=-1)
grid_search.fit(X_train, y_train)
# 输出最佳参数
print(grid_search.best_params_)
'''
#***********************
best_rf = grid_search.best_estimator_  # 获取最优的随机森林模型
y_pred = best_rf.predict(X_test)
print(grid_search.best_params_)
print("accuracy:", accuracy_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
print("roc_auc:", sklearn.metrics.roc_auc_score(y_test,
                                                grid_search.predict_proba(X_test)[:, 1]))
print("f1:", sklearn.metrics.f1_score(y_test, y_pred))
#***************************
'''








