scaler = MinMaxScaler()
for col_name in scores_df.columns:
  scores_df[col_name] = \
    scaler.fit_transform(np.array(scores_df[col_name]).reshape(-1, 1))
scores_df['Avg Score'] = scores_df.sum(axis=1)
