import pandas as pd

# Correct File Path
file_path = '/mnt/e/Student\'s works/MDL articles/GIT article/Sorted_Dataset_GIT.xlsx'

# Load only Column J (Source Column)
df = pd.read_excel(file_path, usecols="J")

# Extract unique values (excluding NaNs)
unique_sources = df.iloc[:, 0].dropna().unique()

# Sort them for easier grouping
unique_sources = sorted(unique_sources)

# Print each unique source
for source in unique_sources:
    print(source)
