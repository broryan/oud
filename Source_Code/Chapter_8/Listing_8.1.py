scores_arr = []
k_vals = [5, 10, 20, 50, 100]
for k in k_vals:
  clf = KNN(n_neighbors=k, method='mean')
  clf.fit(df)
  scores_arr.append(clf.decision_scores_)
s_df = pd.DataFrame(scores_arr)
scores_df['KNN Scores'] = s_df.sum(axis=0)
