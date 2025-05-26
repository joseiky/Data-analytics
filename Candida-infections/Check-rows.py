#!/usr/bin/env python3
import pandas as pd
import sys

def check_rows(xlsx):
    xls = pd.ExcelFile(xlsx, engine='openpyxl')
    total = 0
    print("Rows per sheet:")
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
        n = len(df)
        print(f"  • {sheet!r}: {n:,}")
        total += n
    print(f"\n⇒ Total rows across all sheets: {total:,}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_rows.py YourWorkbook.xlsx")
        sys.exit(1)
    check_rows(sys.argv[1])
