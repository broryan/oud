# This assumes Listing 3.1 has also been run.

tree = BallTree(df)
counts = tree.query_radius(df, 2.0, count_only=True)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))

s = sns.histplot(counts, ax=ax[0]) 
s.set_title(f"Number of Points within Radius")

min_score = min(counts) 
max_score = max(counts)
scores = [(max_score - x)/(max_score - min_score) for x in counts]
s = sns.histplot(scores, ax=ax[1])
s.set_title(f"Scores")

df['Score'] = scores 
threshold = sorted(scores, reverse=True)[15]    
df_flagged = df[df['Score'] >= threshold]
s = sns.scatterplot(data=df, x="A", y="B", ax=ax[2])
s = sns.scatterplot(data=df_flagged, x="A", y="B", color='red', 
                    marker='*', s=200, ax=ax[2])

plt.tight_layout()
plt.show()
