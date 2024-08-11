from mlxtend.frequent_patterns import apriori
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as seaborn

data = fetch_openml('SpeedDating', version=1, parser='auto') 
data_df = pd.DataFrame(data.data, columns=data.feature_names)
data_df = data_df[['d_pref_o_attractive', 'd_pref_o_sincere',  
                   'd_pref_o_intelligence', 'd_pref_o_funny', 
                   'd_pref_o_ambitious', 'd_pref_o_shared_interests']] 
data_df = pd.get_dummies(data_df) 
for col_name in data_df.columns: 
    data_df[col_name] = data_df[col_name].map({0: False, 1: True})
frequent_itemsets = apriori(data_df, min_support=0.3, use_colnames=True) 
