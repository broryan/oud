import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml from sklearn.preprocessing import RobustScaler from sklearn.neighbors import BallTree 
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

data = fetch_openml('abalone', version=1, parser='auto') #A
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.get_dummies(df)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

def score_records():
  scores_df = df.copy()

  clf = IForest() #B
  clf.fit(df)
  scores_df['IF Scores'] = clf.decision_scores_

  # (similar code for LOF, OCSVM, GMM, KDE, HBOS, ECOD, and COPOD 
  #  as for IF)

  clf = LDOFOutlierDetector()
  scores_df['LDOF Scores'] = clf.fit_predict(df, k=5)

  return scores_df

scores_df = score_records()

fig, ax = plt.subplots(figsize=(10, 10))

scores_cols = [x for x in scores_df.columns if "Scores" in x] #C
m = sns.color_palette("Blues", as_cmap=True) #D
sns.heatmap((scores_df[scores_cols].corr(method='spearman')), 
             cmap=m, annot=True)
plt.show()
