from PyNomaly import loop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid", {'axes.grid' : False})

df = create_simple_testdata() 
m = loop.LocalOutlierProbability(df, use_numba=True, 
                                 progress_bar=True).fit() 
scores = m.local_outlier_probabilities
df['LoOP Scores'] = scores 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3)) 
sns.scatterplot(data=df, x='A', y='B', hue='LoOP Scores', 
                size='LoOP Scores', ax=ax[0])
ax[0].legend().remove()

df = create_four_clusters_test_data() 
m = loop.LocalOutlierProbability(df, use_numba=True,
                                 progress_bar=True).fit()
scores = m.local_outlier_probabilities
df['LoOP Scores'] = scores

sns.scatterplot(data=df, x='A', y='B', hue='LoOP Scores', 
                size='LoOP Scores', ax=ax[1])
sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
plt.show()
