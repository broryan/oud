from sklearn.metrics import roc_auc_score

test_df = pd.concat([df, doped_df])
y_true = [0]*len(df) + [1]*len(doped_df)

clf = IForest()
clf.fit(clean_df)
y_pred = clf.decision_function(test_df)
if_auroc = roc_auc_score(y_true, y_pred)

clf = LOF()
clf.fit(clean_df)
y_pred = clf.decision_function(test_df)
lof_auroc = roc_auc_score(y_true, y_pred)
