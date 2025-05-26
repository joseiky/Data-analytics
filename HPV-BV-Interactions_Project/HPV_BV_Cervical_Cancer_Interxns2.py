
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = 'Cleaned HPV-BV data.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# Function to perform PCA and plot
def perform_pca_and_plot(data, features, title):
    scaler = StandardScaler().fit(data[features])
    features_scaled = scaler.transform(data[features])
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    # Explained variance plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained Variance - ' + title)
    plt.show()
    
    # PCA scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PC1', y='PC2', data=principal_df)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of ' + title)
    plt.grid(True)
    plt.show()

# Define the subsets of variables
bacterial_species = ['Fannyhessea vaginae', 'Gardnerella vaginalis', 'Lactobacillus iners', 'Lactobacillus crispatus', 'Lactobacillus jensenii', 'BVAB2', 'Megasphaera 1-2']
hpv_species = ['HPV-16', 'HPV-18', 'HPV-31', 'HPV-33', 'HPV-35', 'HPV-39', 'HPV-45', 'HPV-51', 'HPV-52', 'HPV-56', 'HPV-58', 'HPV-59', 'HPV-68']

# Perform PCA and plot for each subset
perform_pca_and_plot(data, bacterial_species, 'Bacterial Species')
perform_pca_and_plot(data, hpv_species, 'HPV Species')

# Additional subsets based on the specific HPV genotypes combined with bacterial species
subset_3_hpv = ['HPV-16', 'HPV-18', 'HPV-31', 'HPV-35', 'HPV-45', 'HPV-52']
subset_4_hpv = ['HPV-33', 'HPV-39', 'HPV-51', 'HPV-56', 'HPV-58', 'HPV-59', 'HPV-68']

perform_pca_and_plot(data, bacterial_species + subset_3_hpv, 'Subset 3: Bacterial Species and HPV Genotypes 16, 18, 31, 35, 45, 52')
perform_pca_and_plot(data, bacterial_species + subset_4_hpv, 'Subset 4: Bacterial Species and HPV Genotypes 33, 39, 51, 56, 58, 59, 68')
