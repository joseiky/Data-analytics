import os
import pandas as pd

# 1. Load your mapping Excel (first 4 columns):
mapping_df = pd.read_excel('unique_entries_with_interpretation.xlsx', dtype=str)

# Grab the mapping columns by position
old1_col, new1_col, old2_col, new2_col = mapping_df.columns[:4]

# Build lookup dicts
map1 = dict(zip(mapping_df[old1_col], mapping_df[new1_col]))
map2 = dict(zip(mapping_df[old2_col], mapping_df[new2_col]))

INPUT_DIR = 'separate_csvs'

def find_best_match_column(df, lookup_keys):
    """
    Return the column name in df whose values overlap the most
    with lookup_keys. If no overlap, return None.
    """
    best_col = None
    best_count = 0
    lookup_set = set(lookup_keys)
    for col in df.columns:
        # count how many values in this column are in the lookup set
        # dropna() to skip NaNs
        vals = df[col].dropna().astype(str)
        count = vals.isin(lookup_set).sum()
        if count > best_count:
            best_count = count
            best_col = col
    return best_col if best_count > 0 else None

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith('.csv'):
        continue

    path = os.path.join(INPUT_DIR, fname)
    print(f'→ Processing {fname}')
    df = pd.read_csv(path, dtype=str)

    # 2. Auto-detect which column matches map1's old-values
    target1 = find_best_match_column(df, map1.keys())
    if target1:
        df[target1] = df[target1].replace(map1)
        print(f'   • Replaced via map1 in column: {target1}')
    else:
        print(f'   • No column found matching map1 keys.')

    # 3. Auto-detect which column matches map2's old-values
    target2 = find_best_match_column(df, map2.keys())
    if target2:
        df[target2] = df[target2].replace(map2)
        print(f'   • Replaced via map2 in column: {target2}')
    else:
        print(f'   • No column found matching map2 keys.')

    # 4. Save back (overwrite)
    df.to_csv(path, index=False)
    print(f'   ✔ {fname} saved\n')

print('🎉 All done.')
