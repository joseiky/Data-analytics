
import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# Define the output directory
output_dir = '/mnt/e/Student\'s works/MDL articles/HPV-BV article/output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Adjust the path to your dataset
data_path = "Cleaned HPV-BV data.xlsx"

# Load the data from the Excel file, adjust the sheet name if necessary
df = pd.read_excel(data_path, sheet_name='Raw data for Statistics')

# Assuming HPV columns are named as 'HPV_xx' where xx is the type number
# Creating a new column 'Other_HPV' indicating the presence of any HPV type (1 if present, 0 otherwise)
hpv_columns = [col for col in df.columns if 'HPV_' in col]
df['Other_HPV'] = df[hpv_columns].sum(axis=1)
df['Other_HPV'] = (df['Other_HPV'] > 0).astype(int)

# Generating interaction terms for HPV types
df['HPV_16_x_HPV_18'] = df['HPV_16'] * df['HPV_18']
df['HPV_16_x_HPV_31'] = df['HPV_16'] * df['HPV_31']
df['HPV_16_x_HPV_33'] = df['HPV_16'] * df['HPV_33']
df['HPV_16_x_HPV_35'] = df['HPV_16'] * df['HPV_35']
df['HPV_16_x_HPV_39'] = df['HPV_16'] * df['HPV_39']
df['HPV_16_x_HPV_45'] = df['HPV_16'] * df['HPV_45']
df['HPV_16_x_HPV_51'] = df['HPV_16'] * df['HPV_51']
df['HPV_16_x_HPV_52'] = df['HPV_16'] * df['HPV_52']
df['HPV_16_x_HPV_56'] = df['HPV_16'] * df['HPV_56']
df['HPV_16_x_HPV_58'] = df['HPV_16'] * df['HPV_58']
df['HPV_16_x_HPV_59'] = df['HPV_16'] * df['HPV_59']
df['HPV_16_x_HPV_68'] = df['HPV_16'] * df['HPV_68']
df['HPV_18_x_HPV_31'] = df['HPV_18'] * df['HPV_31']
df['HPV_18_x_HPV_33'] = df['HPV_18'] * df['HPV_33']
df['HPV_18_x_HPV_35'] = df['HPV_18'] * df['HPV_35']
df['HPV_18_x_HPV_39'] = df['HPV_18'] * df['HPV_39']
df['HPV_18_x_HPV_45'] = df['HPV_18'] * df['HPV_45']
df['HPV_18_x_HPV_51'] = df['HPV_18'] * df['HPV_51']
df['HPV_18_x_HPV_52'] = df['HPV_18'] * df['HPV_52']
df['HPV_18_x_HPV_56'] = df['HPV_18'] * df['HPV_56']
df['HPV_18_x_HPV_58'] = df['HPV_18'] * df['HPV_58']
df['HPV_18_x_HPV_59'] = df['HPV_18'] * df['HPV_59']
df['HPV_18_x_HPV_68'] = df['HPV_18'] * df['HPV_68']
df['HPV_31_x_HPV_33'] = df['HPV_31'] * df['HPV_33']
df['HPV_31_x_HPV_35'] = df['HPV_31'] * df['HPV_35']
df['HPV_31_x_HPV_39'] = df['HPV_31'] * df['HPV_39']
df['HPV_31_x_HPV_45'] = df['HPV_31'] * df['HPV_45']
df['HPV_31_x_HPV_51'] = df['HPV_31'] * df['HPV_51']
df['HPV_31_x_HPV_52'] = df['HPV_31'] * df['HPV_52']
df['HPV_31_x_HPV_56'] = df['HPV_31'] * df['HPV_56']
df['HPV_31_x_HPV_58'] = df['HPV_31'] * df['HPV_58']
df['HPV_31_x_HPV_59'] = df['HPV_31'] * df['HPV_59']
df['HPV_31_x_HPV_68'] = df['HPV_31'] * df['HPV_68']
df['HPV_33_x_HPV_35'] = df['HPV_33'] * df['HPV_35']
df['HPV_33_x_HPV_39'] = df['HPV_33'] * df['HPV_39']
df['HPV_33_x_HPV_45'] = df['HPV_33'] * df['HPV_45']
df['HPV_33_x_HPV_51'] = df['HPV_33'] * df['HPV_51']
df['HPV_33_x_HPV_52'] = df['HPV_33'] * df['HPV_52']
df['HPV_33_x_HPV_56'] = df['HPV_33'] * df['HPV_56']
df['HPV_33_x_HPV_58'] = df['HPV_33'] * df['HPV_58']
df['HPV_33_x_HPV_59'] = df['HPV_33'] * df['HPV_59']
df['HPV_33_x_HPV_68'] = df['HPV_33'] * df['HPV_68']
df['HPV_35_x_HPV_39'] = df['HPV_35'] * df['HPV_39']
df['HPV_35_x_HPV_45'] = df['HPV_35'] * df['HPV_45']
df['HPV_35_x_HPV_51'] = df['HPV_35'] * df['HPV_51']
df['HPV_35_x_HPV_52'] = df['HPV_35'] * df['HPV_52']
df['HPV_35_x_HPV_56'] = df['HPV_35'] * df['HPV_56']
df['HPV_35_x_HPV_58'] = df['HPV_35'] * df['HPV_58']
df['HPV_35_x_HPV_59'] = df['HPV_35'] * df['HPV_59']
df['HPV_35_x_HPV_68'] = df['HPV_35'] * df['HPV_68']
df['HPV_39_x_HPV_45'] = df['HPV_39'] * df['HPV_45']
df['HPV_39_x_HPV_51'] = df['HPV_39'] * df['HPV_51']
df['HPV_39_x_HPV_52'] = df['HPV_39'] * df['HPV_52']
df['HPV_39_x_HPV_56'] = df['HPV_39'] * df['HPV_56']
df['HPV_39_x_HPV_58'] = df['HPV_39'] * df['HPV_58']
df['HPV_39_x_HPV_59'] = df['HPV_39'] * df['HPV_59']
df['HPV_39_x_HPV_68'] = df['HPV_39'] * df['HPV_68']
df['HPV_45_x_HPV_51'] = df['HPV_45'] * df['HPV_51']
df['HPV_45_x_HPV_52'] = df['HPV_45'] * df['HPV_52']
df['HPV_45_x_HPV_56'] = df['HPV_45'] * df['HPV_56']
df['HPV_45_x_HPV_58'] = df['HPV_45'] * df['HPV_58']
df['HPV_45_x_HPV_59'] = df['HPV_45'] * df['HPV_59']
df['HPV_45_x_HPV_68'] = df['HPV_45'] * df['HPV_68']
df['HPV_51_x_HPV_52'] = df['HPV_51'] * df['HPV_52']
df['HPV_51_x_HPV_56'] = df['HPV_51'] * df['HPV_56']
df['HPV_51_x_HPV_58'] = df['HPV_51'] * df['HPV_58']
df['HPV_51_x_HPV_59'] = df['HPV_51'] * df['HPV_59']
df['HPV_51_x_HPV_68'] = df['HPV_51'] * df['HPV_68']
df['HPV_52_x_HPV_56'] = df['HPV_52'] * df['HPV_56']
df['HPV_52_x_HPV_58'] = df['HPV_52'] * df['HPV_58']
df['HPV_52_x_HPV_59'] = df['HPV_52'] * df['HPV_59']
df['HPV_52_x_HPV_68'] = df['HPV_52'] * df['HPV_68']
df['HPV_56_x_HPV_58'] = df['HPV_56'] * df['HPV_58']
df['HPV_56_x_HPV_59'] = df['HPV_56'] * df['HPV_59']
df['HPV_56_x_HPV_68'] = df['HPV_56'] * df['HPV_68']
df['HPV_58_x_HPV_59'] = df['HPV_58'] * df['HPV_59']
df['HPV_58_x_HPV_68'] = df['HPV_58'] * df['HPV_68']
df['HPV_59_x_HPV_68'] = df['HPV_59'] * df['HPV_68']


# Specify HPV genotype columns and the outcome variable
hpv_genotypes = ['HPV_16', 'HPV_18', 'HPV_31', 'HPV_33', 'HPV_35', 'HPV_39', 'HPV_45', 'HPV_51', 'HPV_52', 'HPV_56', 'HPV_58', 'HPV_59', 'HPV_68']
outcome_variable = 'CERVICAL_CYTOLOGY'

# Building the formula for logistic regression
formula = outcome_variable + " ~ " + " + ".join(hpv_genotypes)

# Fit the logistic regression model
model = formula = 'CERVICAL_CYTOLOGY ~ HPV_16 + HPV_18 + HPV_31 + HPV_33 + HPV_35 + HPV_39 + HPV_45 + HPV_51 + HPV_52 + HPV_56 + HPV_58 + HPV_59 + HPV_68 + HPV_16:HPV_18 + HPV_16:HPV_31 + HPV_16:HPV_33 + HPV_16:HPV_35 + HPV_16:HPV_39 + HPV_16:HPV_45 + HPV_16:HPV_51 + HPV_16:HPV_52 + HPV_16:HPV_56 + HPV_16:HPV_58 + HPV_16:HPV_59 + HPV_16:HPV_68 + HPV_18:HPV_31 + HPV_18:HPV_33 + HPV_18:HPV_35 + HPV_18:HPV_39 + HPV_18:HPV_45 + HPV_18:HPV_51 + HPV_18:HPV_52 + HPV_18:HPV_56 + HPV_18:HPV_58 + HPV_18:HPV_59 + HPV_18:HPV_68 + HPV_31:HPV_33 + HPV_31:HPV_35 + HPV_31:HPV_39 + HPV_31:HPV_45 + HPV_31:HPV_51 + HPV_31:HPV_52 + HPV_31:HPV_56 + HPV_31:HPV_58 + HPV_31:HPV_59 + HPV_31:HPV_68 + HPV_33:HPV_35 + HPV_33:HPV_39 + HPV_33:HPV_45 + HPV_33:HPV_51 + HPV_33:HPV_52 + HPV_33:HPV_56 + HPV_33:HPV_58 + HPV_33:HPV_59 + HPV_33:HPV_68 + HPV_35:HPV_39 + HPV_35:HPV_45 + HPV_35:HPV_51 + HPV_35:HPV_52 + HPV_35:HPV_56 + HPV_35:HPV_58 + HPV_35:HPV_59 + HPV_35:HPV_68 + HPV_39:HPV_45 + HPV_39:HPV_51 + HPV_39:HPV_52 + HPV_39:HPV_56 + HPV_39:HPV_58 + HPV_39:HPV_59 + HPV_39:HPV_68 + HPV_45:HPV_51 + HPV_45:HPV_52 + HPV_45:HPV_56 + HPV_45:HPV_58 + HPV_45:HPV_59 + HPV_45:HPV_68 + HPV_51:HPV_52 + HPV_51:HPV_56 + HPV_51:HPV_58 + HPV_51:HPV_59 + HPV_51:HPV_68 + HPV_52:HPV_56 + HPV_52:HPV_58 + HPV_52:HPV_59 + HPV_52:HPV_68 + HPV_56:HPV_58 + HPV_56:HPV_59 + HPV_56:HPV_68 + HPV_58:HPV_59 + HPV_58:HPV_68 + HPV_59:HPV_68'
smf.mnlogit(formula='CERVICAL_CYTOLOGY ~ AGE + Other_HPV', data=df)
# Combining less prevalent HPV subtypes into 'Other_HPV'
less_prevalent_subtypes = ['HPV_33', 'HPV_18', 'HPV_68']  # Example, adjust based on actual analysis
df['Other_HPV'] = df[less_prevalent_subtypes].max(axis=1)


# Saving the model summary to a text file
summary_path = output_dir + 'model_summary.txt'
with open(summary_path, 'w') as summary_file:
    summary_file.write(model_fit.summary().as_text())

# Extract coefficients and p-values
coefficients = model.params
p_values = model.pvalues

# Prepare the output directory and file names
output_dir = '/mnt/e/Student\'s works/MDL articles/HPV-BV article/output/'
coefficients_csv = output_dir + 'hpv_coefficients_pvalues.csv'
visualization_path = output_dir + 'hpv_coefficients_visualization.png'

# Save to CSV
# Adjusting DataFrame creation to handle scalar and series inputs
if isinstance(coefficients, pd.Series) and isinstance(p_values, pd.Series):
    results_df = pd.DataFrame({'Coefficient': coefficients, 'P-value': p_values})
else:
    results_df = pd.DataFrame({'Coefficient': [coefficients], 'P-value': [p_values]}, index=[0])
results_df.to_csv(coefficients_csv, index=True)

# Visualization
# Filtering only significant coefficients for visualization

# Ensure 'P-value' is numeric
results_df['P-value'] = pd.to_numeric(results_df['P-value'], errors='coerce')

# Filter based on 'P-value'
significant_terms = results_df[results_df['P-value'] < 0.05]

# Group less prevalent HPV types into 'Other_HPV'
less_prevalent_hpv = ['HPV_18', 'HPV_33', 'HPV_58', 'HPV_68']
df['Other_HPV'] = df[less_prevalent_hpv].max(axis=1)

# Adding HPV-HPV interaction terms for dominant types
df['HPV_16_x_HPV_31'] = df['HPV_16'] * df['HPV_31']
df['HPV_16_x_HPV_35'] = df['HPV_16'] * df['HPV_35']
df['HPV_16_x_HPV_39'] = df['HPV_16'] * df['HPV_39']
df['HPV_16_x_HPV_45'] = df['HPV_16'] * df['HPV_45']
df['HPV_16_x_HPV_51'] = df['HPV_16'] * df['HPV_51']
df['HPV_16_x_HPV_52'] = df['HPV_16'] * df['HPV_52']
df['HPV_16_x_HPV_56'] = df['HPV_16'] * df['HPV_56']
df['HPV_16_x_HPV_59'] = df['HPV_16'] * df['HPV_59']
df['HPV_31_x_HPV_35'] = df['HPV_31'] * df['HPV_35']
df['HPV_31_x_HPV_39'] = df['HPV_31'] * df['HPV_39']
df['HPV_31_x_HPV_45'] = df['HPV_31'] * df['HPV_45']
df['HPV_31_x_HPV_51'] = df['HPV_31'] * df['HPV_51']
df['HPV_31_x_HPV_52'] = df['HPV_31'] * df['HPV_52']
df['HPV_31_x_HPV_56'] = df['HPV_31'] * df['HPV_56']
df['HPV_31_x_HPV_59'] = df['HPV_31'] * df['HPV_59']
df['HPV_35_x_HPV_39'] = df['HPV_35'] * df['HPV_39']
df['HPV_35_x_HPV_45'] = df['HPV_35'] * df['HPV_45']
df['HPV_35_x_HPV_51'] = df['HPV_35'] * df['HPV_51']
df['HPV_35_x_HPV_52'] = df['HPV_35'] * df['HPV_52']
df['HPV_35_x_HPV_56'] = df['HPV_35'] * df['HPV_56']
df['HPV_35_x_HPV_59'] = df['HPV_35'] * df['HPV_59']
df['HPV_39_x_HPV_45'] = df['HPV_39'] * df['HPV_45']
df['HPV_39_x_HPV_51'] = df['HPV_39'] * df['HPV_51']
df['HPV_39_x_HPV_52'] = df['HPV_39'] * df['HPV_52']
df['HPV_39_x_HPV_56'] = df['HPV_39'] * df['HPV_56']
df['HPV_39_x_HPV_59'] = df['HPV_39'] * df['HPV_59']
df['HPV_45_x_HPV_51'] = df['HPV_45'] * df['HPV_51']
df['HPV_45_x_HPV_52'] = df['HPV_45'] * df['HPV_52']
df['HPV_45_x_HPV_56'] = df['HPV_45'] * df['HPV_56']
df['HPV_45_x_HPV_59'] = df['HPV_45'] * df['HPV_59']
df['HPV_51_x_HPV_52'] = df['HPV_51'] * df['HPV_52']
df['HPV_51_x_HPV_56'] = df['HPV_51'] * df['HPV_56']
df['HPV_51_x_HPV_59'] = df['HPV_51'] * df['HPV_59']
df['HPV_52_x_HPV_56'] = df['HPV_52'] * df['HPV_56']
df['HPV_52_x_HPV_59'] = df['HPV_52'] * df['HPV_59']
df['HPV_56_x_HPV_59'] = df['HPV_56'] * df['HPV_59']

# Update the model formula to include dominant HPV-HPV interaction terms and 'Other_HPV'
formula = 'CERVICAL_CYTOLOGY ~ AGE + HPV_16 + HPV_31 + HPV_35 + HPV_39 + HPV_45 + HPV_51 + HPV_52 + HPV_56 + HPV_59 + Other_HPV'
