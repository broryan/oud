# This assumes Listing 6.4 has been executed.
# The output is quite large and it may be preferred to view a small
# number of these plots. 

df['HBOS Score'] = pred
df['Outlier'] = (df['HBOS Score'] > df['HBOS Score'].quantile(0.995))

fig, ax = plt.subplots(nrows=20, ncols=20, sharey=True, figsize=(65, 65))

for i in range(20):
  for j in range(20):
    col_name = f"V{(i*20) + j + 1}"
    sns.boxplot(data=df, x=col_name, orient='h', y='Outlier', ax=ax[i][j])
plt.show()
