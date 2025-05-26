
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.formula.api as ols
import os

# Load the dataset
data = pd.read_excel('Cleaned HPV-BV data.xlsx')

# Ensure the output directory exists
os.makedirs('output', exist_ok=True)

def perform_pca_and_save(data, features, file_name_prefix):
    scaler = StandardScaler().fit(data[features])
    features_scaled = scaler.transform(data[features])
    
    pca = PCA(n_components=min(len(features), 2))
    principal_components = pca.fit_transform(features_scaled)
    
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    principal_df.to_csv(f"{file_name_prefix}_principal_components.csv", index=False)
    
    explained_variance_df = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance Ratio'])
    explained_variance_df.to_csv(f"{file_name_prefix}_explained_variance.csv", index=False)

def save_interaction_effects_analysis(data, dependent_var, independent_vars, file_name):
    formula = f"{dependent_var} ~ " + " + ".join(independent_vars) + " + " + " + ".join([f"{x}:{y}" for x in independent_vars for y in independent_vars if x != y])
    model = ols.ols(formula, data=data).fit()
    
    summary_df = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    summary_df.to_csv(file_name)

# Features for analysis
bacterial_species = ['Fannyhessea_vaginae', 'Gardnerella_vaginalis', 'Lactobacillus_iners', 'Lactobacillus_crispatus', 'Lactobacillus_jensenii', 'BVAB2', 'Megasphaera_1_2', 'Lactobacillus_gasseri']
hpv_species_all = ['HPV-16', 'HPV-18', 'HPV-31', 'HPV-33', 'HPV-35', 'HPV-39', 'HPV-45', 'HPV-51', 'HPV-52', 'HPV-56', 'HPV-58', 'HPV-59', 'HPV-68']
hpv_species_subset3 = ['HPV-16', 'HPV-18', 'HPV-31', 'HPV-35', 'HPV-45', 'HPV-52']
hpv_species_subset4 = ['HPV-33', 'HPV-39', 'HPV-51', 'HPV-56', 'HPV-58', 'HPV-59', 'HPV-68']

# Subset 1: Bacterial Species vs. Cervical Cytology and BV
perform_pca_and_save(data, bacterial_species, 'output/subset1_bacterial_species_pca')
save_interaction_effects_analysis(data, 'CERVICAL_CYTOLOGY', bacterial_species, 'output/subset1_bacterial_species_interaction_effects.csv')

# Subset 2: HPV Species vs. Cervical Cytology and BV
perform_pca_and_save(data, hpv_species_all, 'output/subset2_hpv_species_pca')
save_interaction_effects_analysis(data, 'CERVICAL_CYTOLOGY', hpv_species_all, 'output/subset2_hpv_species_interaction_effects.csv')

# Subset 3: Specific Bacterial Species and HPV Genotypes (16, 18, 31, 35, 45, 52)
subset3_features = bacterial_species + hpv_species_subset3
perform_pca_and_save(data, subset3_features, 'output/subset3_combined_pca')
save_interaction_effects_analysis(data, 'CERVICAL_CYTOLOGY', subset3_features, 'output/subset3_combined_interaction_effects.csv')

# Subset 4: Specific Bacterial Species and HPV Genotypes (33, 39, 51, 56, 58, 59, 68)
subset4_features = bacterial_species + hpv_species_subset4
perform_pca_and_save(data, subset4_features, 'output/subset4_combined_pca')
save_interaction_effects_analysis(data, 'CERVICAL_CYTOLOGY', subset4_features, 'output/subset4_combined_interaction_effects.csv')
