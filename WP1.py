import pandas as pd
import numpy as np
import os

# === WP-1: Full-data preprocessing script ===

# 1. Configuration
# get the folder where this script lives:
script_dir = os.path.dirname(os.path.abspath(__file__))
# point data_dir at the separate_csvs subfolder:
data_dir = os.path.join(script_dir, 'separate_csvs')
files = [
    'File 1.csv',
    'File 2.csv',
    'File 3.csv',
    'File 4.csv',
    'File 5.csv',
    'File 6-NY.csv',
    'Resistance.xlsx'
]

# Age group bins and labels
age_bins = [0, 17, 24, 34, 44, 54, 200]
age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '≥55']

# 2. Read & concatenate all parts
frames = []
for fname in files:
    path = os.path.join(data_dir, fname)
    if fname.lower().endswith('.csv'):
        df = pd.read_csv(path, dtype=str)
    else:  # Excel
        df = pd.read_excel(path, dtype=str)
    # Drop fully empty rows
    df = df.dropna(how='all')
    frames.append(df)
master = pd.concat(frames, ignore_index=True)

# 3. Harmonise & parse key fields

# Combine Test Name and Test Subtype when they differ
master['ResolvedTest'] = np.where(
    master['Test Name'].str.strip().str.upper() == master['Test Subtype'].str.strip().str.upper(),
    master['Test Name'],
    master['Test Name'] + ' → ' + master['Test Subtype']
)

# Parse collection date
master['CollectionDate'] = pd.to_datetime(
    master['Date Specimen Collected'], errors='coerce'
)
master['Year']  = master['CollectionDate'].dt.year
master['Month'] = master['CollectionDate'].dt.month

# Bin patient age
master['Pt AGE'] = pd.to_numeric(master['Pt AGE'], errors='coerce')
master['AgeGroup'] = pd.cut(
    master['Pt AGE'], bins=age_bins, labels=age_labels, right=True
)

# 4a. Build obs_table (one row per test observation)
obs_table = master[[
    'MDLNo', 'MDL Patient ID', 'CollectionDate',
    'Specimen', 'Source', 'BV-Status', 'Year', 'Month',
    'ResolvedTest', 'Test Result', 'Subtype Concentration',
    'Pt AGE', 'Pt Ethnicity'
]].rename(columns={
    'Test Result': 'Result',
    'Subtype Concentration': 'Concentration'
})

# Flag positive results (non-null & not 'N')
obs_table['is_pos'] = obs_table['Result'].notna() & (obs_table['Result'] != 'N')

# 4b. Build sample_table (wide pivot by MDLNo)

# Positivity flags
flag_df = obs_table.pivot_table(
    index='MDLNo',
    columns='ResolvedTest',
    values='is_pos',
    aggfunc='max',
    fill_value=False
)

# Mean concentration per sample
conc_df = obs_table.pivot_table(
    index='MDLNo',
    columns='ResolvedTest',
    values='Concentration',
    aggfunc=lambda x: pd.to_numeric(x, errors='coerce').mean()
)

# Combine flags and concentrations
sample_table = pd.concat([flag_df, conc_df.add_suffix('_conc')], axis=1)

# Attach metadata back to sample_table
meta = master.drop_duplicates('MDLNo')[[
    'MDLNo', 'MDL Patient ID', 'Specimen', 'Source', 'BV-Status',
    'CollectionDate', 'Year', 'Month', 'AgeGroup', 'Pt Ethnicity'
]].set_index('MDLNo')

sample_table = sample_table.join(meta, how='left')

# 5. Export processed tables
out_dir = os.path.join(data_dir, 'processed')
os.makedirs(out_dir, exist_ok=True)

master.to_parquet(os.path.join(out_dir, 'master.parquet'), index=False)
obs_table.to_parquet(os.path.join(out_dir, 'obs_table.parquet'), index=False)
sample_table.to_parquet(os.path.join(out_dir, 'sample_table.parquet'), index=False)

print("WP-1 preprocessing complete.")
print(f" master shape      : {master.shape}")
print(f" obs_table shape   : {obs_table.shape}")
print(f" sample_table shape: {sample_table.shape}")
