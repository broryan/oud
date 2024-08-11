top_scores_df = scores_df[scores_cols].copy()
for col_name in top_scores_df.columns: 
  top_scores_df[col_name] = top_scores_df[col_name].rank()
  top_scores_df[col_name] = top_scores_df[col_name].apply(
      lambda x: x if x > (len(df) * 0.95) else 0.0)
