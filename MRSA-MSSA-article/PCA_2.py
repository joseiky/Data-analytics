import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Data Loading and Encoding ---
df = pd.read_csv('Dataset_1_Cleaned_masterfile.csv', dtype=str)

# Numeric columns
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Microbial/pathology outcomes as binary
binary_factors = [
    'S. aureus', 'MRSA', 'MSSA', 'NILM', 'ECA', 'LSIL', 'HSIL', 'ASCUS',
    'TRICH', 'CD', 'BV', 'Actino', 'Atrophic VAG'
]
for factor in binary_factors:
    if factor in df.columns:
        df[factor] = (df[factor] == '1').astype(int)

# BV-Status dummies
if 'BV-Status' in df.columns:
    bv_dummies = pd.get_dummies(df['BV-Status'], prefix='BV')
    for col in bv_dummies.columns:
        df[col] = bv_dummies[col]

# Gender/Sex/Ethnicity dummies
for cat in ['Gender', 'Sex', 'Ethnicity']:
    if cat in df.columns:
        dummies = pd.get_dummies(df[cat], prefix=cat)
        df = pd.concat([df, dummies], axis=1)

# Select columns for PCA and clustering
corr_cols = (
    ['Age'] +
    [c for c in df.columns if c.startswith('BV_')] +
    [f for f in binary_factors if f in df.columns] +
    [c for c in df.columns if c.startswith('Gender_') or c.startswith('Sex_') or c.startswith('Ethnicity_')]
)
corr_df = df[corr_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
corr_df_1d = corr_df[[c for c in corr_df.columns if corr_df[c].ndim == 1]]

# --- PCA and K-means clustering ---
pca_input = corr_df_1d.dropna()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(pca_input)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(pca_input)
# Save plotting data for export
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'KMeansCluster': clusters
})
if 'BV-Status' in df.columns:
    pca_df['BV-Status'] = df.loc[pca_input.index, 'BV-Status'].values
pca_df.to_csv('PCA_KMeans_plotting_data.csv', index=False)

# --- PCA/K-means Multi-panel Plot ---
fig, axs = plt.subplots(1, 2, figsize=(16, 7))
scatter = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6, s=30)
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")
axs[0].set_title("PCA colored by K-means cluster")
plt.colorbar(scatter, ax=axs[0], label='K-means cluster')
if 'BV-Status' in df.columns:
    unique_labels = pd.Categorical(pca_df['BV-Status']).codes
    scatter2 = axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=unique_labels, cmap='Set1', alpha=0.6, s=30)
    axs[1].set_xlabel("PC1")
    axs[1].set_ylabel("PC2")
    axs[1].set_title("PCA colored by BV-Status")
    plt.colorbar(scatter2, ax=axs[1], ticks=range(len(set(pca_df['BV-Status']))), label='BV-Status')
plt.suptitle("PCA and K-means Clustering (Multi-panel)", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('PCA_KMeans_multipanel.png', dpi=400)
plt.show()

# --- Within-patient Variation Analysis (Multi-panel) ---
id_col = 'Patient-ID'
counts = df[id_col].value_counts()
multi_patients = counts[counts > 1].index
within_df = df[df[id_col].isin(multi_patients)].copy()
within_df['S_aureus_conc'] = pd.to_numeric(within_df.get('Concentration', np.nan), errors='coerce')
within_df['BV_P'] = (within_df['BV-Status'] == 'BV-P').astype(int)
within_df['BV_T'] = (within_df['BV-Status'] == 'BV-T').astype(int)
if 'ECA' in within_df.columns:
    within_df['ECA_pos'] = (within_df['ECA'] == 1).astype(int)

# Save within-patient plotting data
within_df[['Patient-ID', 'S_aureus_conc', 'BV_P', 'BV_T', 'ECA_pos']].to_csv('Within_Patient_plotting_data.csv', index=False)

# Prepare summary data for boxplots
patient_bv = within_df.groupby(id_col)[['BV_P', 'BV_T']].mean().melt(ignore_index=False).reset_index()
if 'ECA_pos' in within_df.columns:
    patient_eca = within_df.groupby(id_col)['ECA_pos'].mean()
else:
    patient_eca = pd.Series([])

# Multi-panel figure for within-patient variations
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
# S. aureus concentration variation
sns.boxplot(x=id_col, y='S_aureus_conc', data=within_df, showfliers=False, ax=axs[0])
axs[0].set_title("Within-patient Variation\nS. aureus Concentration")
axs[0].set_xlabel("Patient ID")
axs[0].set_ylabel("S. aureus Conc.")
axs[0].set_xticks([])
# BV-Positive/Transitional proportion
sns.boxplot(x='variable', y='value', data=patient_bv, showfliers=False, ax=axs[1])
axs[1].set_title("Within-patient Proportion\nBV-Positive / BV-Transitional")
axs[1].set_xlabel("BV State")
axs[1].set_ylabel("Proportion of Samples")
# ECA positive
if len(patient_eca) > 0:
    sns.boxplot(y=patient_eca, showfliers=False, ax=axs[2])
    axs[2].set_title("Within-patient Proportion\nECA-Positive")
    axs[2].set_ylabel("Proportion of ECA+")
else:
    axs[2].set_visible(False)
fig.suptitle("Within-Patient Variation (Multi-panel)", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('WithinPatient_multipanel.png', dpi=400)
plt.show()
