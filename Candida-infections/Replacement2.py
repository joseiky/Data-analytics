import os
import pandas as pd

# 1) Load your mapping
mapping_df = pd.read_excel('unique_entries_with_interpretation.xlsx', dtype=str)
old1_col, new1_col, old2_col, new2_col = mapping_df.columns[:4]
map1 = dict(zip(mapping_df[old1_col], mapping_df[new1_col]))
map2 = dict(zip(mapping_df[old2_col], mapping_df[new2_col]))

INPUT_DIR = 'separate_csvs'
EXTS = ('.csv', '.xlsx')

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(EXTS):
        continue

    path = os.path.join(INPUT_DIR, fname)
    print(f"\n→ Opening {fname}")

    # 2) Read in
    if fname.lower().endswith('.csv'):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)

    changed = False

    # 3) ONLY touch Source
    if 'Source' in df.columns:
        before = df['Source'].copy()
        df['Source'] = df['Source'].replace(map1)
        if not df['Source'].equals(before):
            print(f"   • Source: { (before != df['Source']).sum() } replacements")
            changed = True

    # 4) ONLY touch Pt Ethnicity
    if 'Pt Ethnicity' in df.columns:
        before = df['Pt Ethnicity'].copy()
        df['Pt Ethnicity'] = df['Pt Ethnicity'].replace(map2)
        if not df['Pt Ethnicity'].equals(before):
            print(f"   • Pt Ethnicity: { (before != df['Pt Ethnicity']).sum() } replacements")
            changed = True

    # 5) Skip if no change
    if not changed:
        print("   • No relevant columns or no changes, skipping.")
        continue

    # 6) Write back
    if fname.lower().endswith('.csv'):
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    print(f"   ✔ {fname} saved.")

print("\n🎉 Done.")
