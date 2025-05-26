#!/usr/bin/env python3
import os
import glob
import pandas as pd
import sys

def find_uniques_combined(input_folder, output_excel):
    uniques_C = set()
    uniques_M = set()

    # Gather file paths
    csv_paths  = glob.glob(os.path.join(input_folder, '*.csv'))
    xlsx_paths = glob.glob(os.path.join(input_folder, '*.xlsx'))

    def collect(df, name):
        cols = df.columns
        if len(cols) >= 3:
            uniques_C.update(df.iloc[:,2].dropna().astype(str).unique())
        else:
            print(f"[!] {name}: <3 columns, skipping C")
        if len(cols) >= 13:
            uniques_M.update(df.iloc[:,12].dropna().astype(str).unique())
        else:
            print(f"[!] {name}: <13 columns, skipping M")

    # Process CSVs
    for p in csv_paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        collect(df, os.path.basename(p))

    # Process Excel files
    for p in xlsx_paths:
        df = pd.read_excel(p, dtype=str, engine='openpyxl')
        collect(df, os.path.basename(p))

    # Prepare lists and pad
    list_C = sorted(uniques_C)
    list_M = sorted(uniques_M)
    max_len = max(len(list_C), len(list_M))
    list_C += [''] * (max_len - len(list_C))
    list_M += [''] * (max_len - len(list_M))

    df_out = pd.DataFrame({
        'Unique_Column_C': list_C,
        'Unique_Column_M': list_M
    })

    # Write to a single-sheet Excel
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False, sheet_name='Uniques')

    print(f"✅ Wrote {max_len} unique entries to '{output_excel}' in sheet 'Uniques'")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python find_uniques_combined.py <input_folder> <output.xlsx>")
        sys.exit(1)
    find_uniques_combined(sys.argv[1], sys.argv[2])
