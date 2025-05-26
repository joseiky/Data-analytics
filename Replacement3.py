import os
import pandas as pd

# 1) Load the original mapping
mapping_df = pd.read_excel('unique_entries_with_interpretation.xlsx', dtype=str)
old1, new1, old2, new2 = mapping_df.columns[:4]
map1 = dict(zip(mapping_df[old1], mapping_df[new1]))
map2 = dict(zip(mapping_df[old2], mapping_df[new2]))

FILE = 'separate_csvs/Resistance.xlsx'

# 2) Read the Excel
df = pd.read_excel(FILE, dtype=str)

# 3) Drop any fully-blank columns and any 'Unnamed' spurious headers
df = df.dropna(axis=1, how='all')               # drop columns where every cell is NaN
df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]  # drop columns whose name starts with 'Unnamed'

# 4) Re-apply **only** the valid replacements:

#   Source
if 'Source' in df:
    df['Source'] = df['Source'].replace(map1)

#   Pt Ethnicity
if 'Pt Ethnicity' in df:
    df['Pt Ethnicity'] = df['Pt Ethnicity'].replace(map2)

# 5) Save back (overwrites Resistance.xlsx)
df.to_excel(FILE, index=False)
print("✅ Resistance.xlsx cleaned and Source/Pt Ethnicity re-mapped.")
