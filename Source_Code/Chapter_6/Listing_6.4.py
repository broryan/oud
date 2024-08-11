import pandas as pd
from pyod.models.hbos import HBOS
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_openml("speech", version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
display(df.head())

det = HBOS()
det.fit(df)
pred = det.decision_scores_

sns.histplot(pred)
plt.show()
