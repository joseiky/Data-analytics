#!/usr/bin/env python3
"""
merge_ml_feature_outputs.py

Merges all sheets from:
  - Merged_ML_Outputs.xlsx
  - Feature_Outputs.xlsx

into Combined_Outputs.xlsx, preserving sheet names
(and appending _1, _2… on any name collisions).
"""

import pandas as pd

# Input filenames
file1 = "Merged_ML_Outputs.xlsx"
file2 = "Feature_Outputs.xlsx"
out_file = "Combined_Outputs.xlsx"

def merge_excel_files(f1, f2, out):
    # Read all sheets from each file
    xls1 = pd.read_excel(f1, sheet_name=None)
    xls2 = pd.read_excel(f2, sheet_name=None)

    # Combine, handling duplicate sheet names
    merged = {}
    merged.update(xls1)
    for name, df in xls2.items():
        new_name = name
        if new_name in merged:
            # find a free name suffix
            suffix = 1
            while f"{new_name}_{suffix}" in merged:
                suffix += 1
            new_name = f"{new_name}_{suffix}"
        merged[new_name] = df

    # Write out all sheets into one file
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        for sheet_name, df in merged.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Merged {len(xls1)} sheets from '{f1}' and {len(xls2)} sheets from '{f2}'")
    print(f"   into {len(merged)} sheets in '{out}'")

if __name__ == "__main__":
    merge_excel_files(file1, file2, out_file)
