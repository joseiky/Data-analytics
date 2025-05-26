import pandas as pd

# 1. Load only the MDLNo column
df = pd.read_excel('Complete_Dataset_GIT.xlsx', usecols=['MDLNo'])

# 2. Total number of rows (samples)
total_rows = len(df)

# 3. Count how many times each MDLNo appears
freq = df['MDLNo'].value_counts()

# 4. Number of unique MDLNos
unique_mdlnos = freq.size

# 5. Number (and percentage) of MDLNos that appear exactly once
single = (freq == 1).sum()
single_pct = single / unique_mdlnos * 100

# 6. Number (and percentage) of MDLNos that appear more than once
multiple = (freq > 1).sum()
multiple_pct = multiple / unique_mdlnos * 100

print(f"Total rows: {total_rows}")
print(f"Unique MDLNos: {unique_mdlnos}")
print(f"  • Appear only once: {single} ({single_pct:.1f}%)")
print(f"  • Appear > once:    {multiple} ({multiple_pct:.1f}%)")
