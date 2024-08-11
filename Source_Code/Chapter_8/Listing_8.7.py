from sklearn.preprocessing import RobustScaler

top_scores_df = scores_df[scores_cols].copy()
for col_name in top_scores_df.columns:
  scaler = RobustScaler()
  top_scores_df[col_name] = scaler.fit_transform(
      np.array(top_scores_df[col_name]).reshape(-1, 1))
  top_scores_df[col_name] =\
   top_scores_df[col_name].apply(lambda x: x if x > 2.0 else 0.0)

fig, ax = plt.subplots(figsize=(10, 10))
m = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(top_scores_df.corr(method='spearman'), cmap=m, annot=True)
plt.show()
