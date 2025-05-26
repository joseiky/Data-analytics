#!/usr/bin/env python3
import sys
import os
import pandas as pd

def sanitize(filename: str) -> str:
    """
    Turn any characters not alphanumeric, space, underscore or hyphen
    into underscores, and strip leading/trailing whitespace.
    """
    return "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_"
        for c in filename
    ).strip()

def export_sheets(input_xlsx: str, output_dir: str):
    # create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # load workbook
    xls = pd.ExcelFile(input_xlsx, engine="openpyxl")

    # export each sheet
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl", dtype=str)
        csv_name = sanitize(sheet) + ".csv"
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_name}: {len(df):,} rows × {df.shape[1]:,} cols")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_sheets.py Dataset-filtered.xlsx output_folder")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_folder = sys.argv[2]
    export_sheets(input_file, output_folder)
