import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

df = pd.read_excel('/mnt/e/Student\'s works/MDL articles/HPV-BV article/Cleaned HPV-BV data.xlsx', sheet_name=0)

df_clustering = df.drop(columns=['Sample #', 'AGE', 'BV Status', 'CERVICAL CYTOLOGY\'\''])

# For the full dataset (may be computationally intensive):
linkage_matrix = linkage(df_clustering, method='ward')

# For a sampled subset (replace 500 with your desired sample size):
df_sample = df_clustering.sample(n=500, random_state=42)
linkage_matrix_sample = linkage(df_sample, method='ward')

# For the full dataset:
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram (Full Dataset)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# For the sampled subset:
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix_sample)
plt.title('Hierarchical Clustering Dendrogram (Sampled Subset)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
