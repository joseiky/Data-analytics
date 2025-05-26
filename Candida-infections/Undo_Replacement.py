import os
import pandas as pd

# 1) Load the same mapping file you used before
mapping_df = pd.read_excel('unique_entries_with_interpretation.xlsx', dtype=str)
old1_col, new1_col, old2_col, new2_col = mapping_df.columns[:4]

# 2) Build reverse lookup dicts (new_value → old_value)
rev_map1 = {new: old for old, new in zip(mapping_df[old1_col], mapping_df[new1_col])}
rev_map2 = {new: old for old, new in zip(mapping_df[old2_col], mapping_df[new2_col])}

INPUT_DIR = 'separate_csvs'
EXTS = ('.csv', '.xlsx')

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(EXTS):
        continue

    path = os.path.join(INPUT_DIR, fname)
    print(f'\n→ Reverting {fname}')

    # 3) Read file
    if fname.lower().endswith('.csv'):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)

    changed = False

    # 4) Undo map1 on the Specimen column (if it exists)
    if 'Specimen' in df.columns:
        before = df['Specimen'].copy()
        df['Specimen'] = df['Specimen'].replace(rev_map1)
        n = (before != df['Specimen']).sum()
        if n:
            print(f'   • Specimen: reverted {n} cells')
            changed = True

    # 5) Undo map2 on the Subtype Result column (if it exists)
    if 'Subtype Result' in df.columns:
        before = df['Subtype Result'].copy()
        df['Subtype Result'] = df['Subtype Result'].replace(rev_map2)
        n = (before != df['Subtype Result']).sum()
        if n:
            print(f'   • Subtype Result: reverted {n} cells')
            changed = True

    if not changed:
        print('   • Nothing to revert here, skipping.')
        continue

    # 6) Write back (overwrite)
    if fname.lower().endswith('.csv'):
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    print(f'   ✔ {fname} reverted.')

print('\n✅ Undo complete.')
