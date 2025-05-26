#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WP-6: PCA & Clustering of GIT Pathogen Profiles
Uses GIT-No-toxins.xlsx both for pathogens and metadata.
Produces Figures 6A–6D, scree plot, biplot, and silhouette scores.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data (pathogens + metadata)
data = pd.read_excel('GIT-No-toxins.xlsx')

# 2. Build binary Sample×Pathogen matrix (only positives)
df_pos = data[data['Result'] == 'P']
mat = pd.crosstab(df_pos['MDLNo'], df_pos['Test']).astype(int)

# 3. Extract metadata per MDLNo (first row for each)
meta = (
    data
    .loc[:, ['MDLNo', 'Age', 'Ethnicity', 'Gender', 'BV-Status', 'Specimen', 'Source']]
    .drop_duplicates(subset='MDLNo')
    .set_index('MDLNo')
)

# Add pathogen richness
meta['richness'] = mat.sum(axis=1)

# 4. PCA on binary matrix
X = mat.values
pca = PCA(n_components=3, random_state=0)
pcs = pca.fit_transform(X)
explained = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(6,4), dpi=300)
plt.plot(np.arange(1,4), explained*100, '-o', lw=2)
plt.xticks([1,2,3])
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.title('WP6 Scree Plot')
plt.tight_layout()
plt.savefig('WP6_scree.png', dpi=300)

# Biplot (PC1 vs PC2)
plt.figure(figsize=(6,6), dpi=300)
sns.scatterplot(x=pcs[:,0], y=pcs[:,1], s=30, alpha=0.7)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
for i, pathogen in enumerate(mat.columns[:10]):
    plt.arrow(0, 0, loadings[i,0]*5, loadings[i,1]*5, color='red', alpha=0.5)
    plt.text(loadings[i,0]*5*1.1, loadings[i,1]*5*1.1, pathogen, fontsize=8)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('WP6 PCA Biplot')
plt.tight_layout()
plt.savefig('WP6_biplot.png', dpi=300)

# 5. Prepare PCA + metadata DataFrame
pca_df = pd.DataFrame(pcs[:,:2], index=mat.index, columns=['PC1','PC2'])
pca_df = pca_df.join(meta)

# 6. Clustering (e.g., K=4)
k = 4
labels = KMeans(n_clusters=k, random_state=0).fit_predict(pcs[:,:2])
pca_df['cluster'] = labels

# Silhouette score for k=4
sil_score = silhouette_score(pcs[:,:2], labels)
print(f"Silhouette score (k={k}): {sil_score:.3f}")

# 7. Plot panels

sns.set(style='whitegrid', font_scale=1.0)

# 6A: PCA colored by richness
plt.figure(figsize=(6,6), dpi=300)
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='richness', palette='viridis', s=40, edgecolor='k')
plt.title('Figure 6A: PCA by Pathogen Richness')
plt.legend(title='Richness', bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig('Figure6A.png', dpi=300)

# 6B: PCA colored by cluster
plt.figure(figsize=(6,6), dpi=300)
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='tab10', s=40, edgecolor='k')
plt.title('Figure 6B: PCA by Cluster')
plt.legend(title='Cluster', bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig('Figure6B.png', dpi=300)

# 6C: PCA overlay BV-Status (color) & Gender (shape)
plt.figure(figsize=(6,6), dpi=300)
sns.scatterplot(
    data=pca_df, x='PC1', y='PC2',
    hue='BV-Status', style='Gender',
    s=50
)
plt.title('Figure 6C: PCA with BV Status & Gender')
plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig('Figure6C.png', dpi=300)

# 6D: Heatmap of cluster vs. BV-Status
heat = pd.crosstab(pca_df['cluster'], pca_df['BV-Status'])
plt.figure(figsize=(6,4), dpi=300)
sns.heatmap(heat, annot=True, fmt='d', cbar=False)
plt.title('Figure 6D: Cluster × BV Status')
plt.ylabel('Cluster'); plt.xlabel('BV Status')
plt.tight_layout()
plt.savefig('Figure6D.png', dpi=300)

# 8. Print cluster enrichment tables
for var in ['BV-Status','Specimen','Source','Ethnicity']:
    print(f"\nCluster vs {var}:\n", pd.crosstab(pca_df['cluster'], pca_df[var]))
