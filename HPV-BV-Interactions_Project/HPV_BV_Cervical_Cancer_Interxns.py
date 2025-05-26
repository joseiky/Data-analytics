
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'Cleaned HPV-BV data.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# Define dependent variable and independent variables
dependent_var = 'CERVICAL CYTOLOGY'
independent_vars = data.columns[3:-1]  # Excludes Sample #, AGE, BV Status, and the dependent variable

# Interaction Effects Analysis
formula = f"{dependent_var} ~ " + " + ".join(independent_vars) + " + " + " + ".join([f"{var}*{hpv}" for var in independent_vars[:8] for hpv in independent_vars[8:]])
model = ols(formula, data=data).fit()
interaction_effects_results = model.summary()
print(interaction_effects_results)

# Visualize the coefficients of interaction terms (for significant interactions or a subset)
fig, ax = plt.subplots(figsize=(10, 6))
coefs = model.params[1:]  # Excluding the intercept
coefs = coefs.sort_values()
coefs.plot(kind='bar', ax=ax)
plt.title('Coefficients of Interaction Terms')
plt.xlabel('Terms')
plt.ylabel('Coefficient Value')
plt.tight_layout()
plt.show()

# PCA Visualization
X_std = StandardScaler().fit_transform(data[independent_vars])
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_std)

# Assign different colors to bacterial species and HPV genotypes
colors = ['blue'] * 8 + ['red'] * (len(independent_vars) - 8)  # Blue for bacterial species, Red for HPV genotypes
plt.figure(figsize=(8, 6))
for i, color in enumerate(colors):
    plt.scatter(principalComponents[:, 0][data[independent_vars.iloc[:, i]] == 1], principalComponents[:, 1][data[independent_vars.iloc[:, i]] == 1], color=color, alpha=0.5, label=independent_vars[i] if i < 9 else None)
plt.title('PCA of Bacterial Species and HPV Genotypes with Color Coding')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
