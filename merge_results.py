#!/usr/bin/env python3
import sys
import pandas as pd

def main(input_xlsx, output_csv):
    # 1) Open the Excel file
    xls = pd.ExcelFile(input_xlsx, engine='openpyxl')

    dfs = []
    expected_total = 0

    # 2) Read each sheet, count rows, and collect
    print("Reading sheets and counting rows:")
    for sheet in xls.sheet_names:
        # read entire sheet as strings to avoid mixed‐type truncation
        df = pd.read_excel(xls,
                           sheet_name=sheet,
                           engine='openpyxl',
                           dtype=str)
        n_rows = len(df)
        print(f"  • {sheet!r:10}: {n_rows:,} rows")
        expected_total += n_rows
        dfs.append(df)

    print(f"\nExpected total rows (sum of sheets): {expected_total:,}\n")

    # 3) Concatenate all DataFrames
    combined = pd.concat(dfs, ignore_index=True)
    actual_total = len(combined)
    print(f"Combined DataFrame has:             {actual_total:,} rows\n")

    # 4) Write out to CSV
    combined.to_csv(output_csv, index=False)
    print(f"Wrote concatenated CSV to:          {output_csv!r}\n")

    # 5) Quick sanity‐check: reopen and count
    reopened = pd.read_csv(output_csv, usecols=[0])  # just one column to speed up
    reopened_count = len(reopened)
    print(f"Reopened CSV reports:               {reopened_count:,} rows (minus header)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python debug_merge.py input.xlsx output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
