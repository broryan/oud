def test_training_size(n_rows):
  clf = LOF()
  clf.fit(clean_df.sample(n=n_rows))
  y_pred = clf.decision_function(test_df)
  lof_auroc = roc_auc_score(y_true, y_pred)
  return lof_auroc
