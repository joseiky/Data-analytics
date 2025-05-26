import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

# Adjust the path to your dataset
data_path = "Cleaned HPV-BV data.xlsx"

# Load the data from the Excel file, adjust the sheet name if necessary
df = pd.read_excel(data_path, sheet_name='Raw data for Statistics')

# Specify HPV genotype columns and the outcome variable
hpv_genotypes = ['HPV_16', 'HPV_18', 'HPV_31', 'HPV_33', 'HPV_35', 'HPV_39', 'HPV_45', 'HPV_51', 'HPV_52', 'HPV_56', 'HPV_58', 'HPV_59', 'HPV_68']
outcome_variable = 'CERVICAL_CYTOLOGY'

# Building the formula for logistic regression
formula = outcome_variable + " ~ " + " + ".join(hpv_genotypes)

# Fit the logistic regression model
model = smf.mnlogit(formula=formula, data=df).fit()

# Saving the model summary to a text file
summary_path = output_dir + 'model_summary.txt'
with open(summary_path, 'w') as summary_file:
    summary_file.write(model.summary().as_text())

# Extract coefficients and p-values
coefficients = model.params
p_values = model.pvalues

# Prepare the output directory and file names
output_dir = '/mnt/e/Student\'s works/MDL articles/HPV_BV article/output/'
coefficients_csv = output_dir + 'hpv_coefficients_pvalues.csv'
visualization_path = output_dir + 'hpv_coefficients_visualization.png'

# Save to CSV
results_df = pd.DataFrame({'Coefficient': coefficients, 'P-value': p_values})
results_df.to_csv(coefficients_csv, index=True)

# Visualization
plt.figure(figsize=(10, 8))
# Filtering only significant coefficients for visualization
significant_terms = results_df[results_df['P-value'] < 0.05]
plt.barh(significant_terms.index, significant_terms['Coefficient'], color='green')
plt.xlabel('Coefficient')
plt.title('Significant HPV Genotype Coefficients')
plt.tight_layout()
plt.savefig(visualization_path)
plt.show()
