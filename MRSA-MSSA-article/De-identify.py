#!/usr/bin/env python3
"""
De-identify MDLNo and Patient-ID in these two specific files:
  • Table S1. Dataset-MRSA-MSSA.xlsx
  • Table S2. Duplicated_MDLNo_Records.xlsx

Scans all sheets in both files, creates sequential codes for all unique MDLNo and Patient-ID,
and outputs:
  - MDL_mapping.csv
  - Patient_mapping.csv
  - Table S1. Dataset-MRSA-MSSA_deidentified.xlsx
  - Table S2. Duplicated_MDLNo_Records_deidentified.xlsx

Usage:
  python De-identify.py
"""

import os
import pandas as pd

# === STEP 1: Define your two specific files ===
file_list = [
    "Table S1. Dataset-MRSA-MSSA.xlsx",
    "Table S2. Duplicated_MDLNo_Records.xlsx"
]

for f in file_list:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

# === STEP 2: Scan all sheets in both files for unique IDs ===
all_mdl = set()
all_pat = set()

for filepath in file_list:
    xlsx = pd.ExcelFile(filepath, engine="openpyxl")
    for sheet in xlsx.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet, dtype=str)
        if "MDLNo" in df.columns:
            all_mdl.update(df["MDLNo"].dropna().astype(str).unique())
        if "Patient-ID" in df.columns:
            all_pat.update(df["Patient-ID"].dropna().astype(str).unique())

print(f"\n• Found {len(all_mdl)} unique MDLNo values")
print(f"• Found {len(all_pat)} unique Patient-ID values")

# === STEP 3: Create mapping dictionaries (orig → new code) ===
mdl_map = {
    orig: f"MDL{str(i).zfill(5)}"
    for i, orig in enumerate(sorted(all_mdl), start=1)
}
pat_map = {
    orig: f"PAT{str(i).zfill(5)}"
    for i, orig in enumerate(sorted(all_pat), start=1)
}

# === STEP 4: Write out mapping files ===
pd.DataFrame({
    "MDLNo_orig": list(mdl_map.keys()),
    "MDLNo_new": list(mdl_map.values())
}).to_csv("MDL_mapping.csv", index=False)
print("✓ Wrote MDL_mapping.csv")

pd.DataFrame({
    "PatientID_orig": list(pat_map.keys()),
    "PatientID_new": list(pat_map.values())
}).to_csv("Patient_mapping.csv", index=False)
print("✓ Wrote Patient_mapping.csv")

# === STEP 5: De-identify both workbooks and save ===
for original_path in file_list:
    print(f"\n→ De-identifying '{original_path}' …")
    xlsx = pd.ExcelFile(original_path, engine="openpyxl")
    output_path = original_path.replace(".xlsx", "_deidentified.xlsx")
    writer = pd.ExcelWriter(output_path, engine="openpyxl")

    for sheet in xlsx.sheet_names:
        df = pd.read_excel(original_path, sheet_name=sheet, dtype=str)
        # Replace MDLNo and Patient-ID if present
        if "MDLNo" in df.columns:
            df["MDLNo"] = df["MDLNo"].map(lambda v: mdl_map.get(str(v), v))
        if "Patient-ID" in df.columns:
            df["Patient-ID"] = df["Patient-ID"].map(lambda v: pat_map.get(str(v), v))
        df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"  • Processed sheet '{sheet}' (rows: {len(df)})")

    writer.close()
    print(f"✓ Saved de-identified file: {os.path.basename(output_path)}")

print("\nAll done! Submit the *_deidentified.xlsx files and mapping CSVs.")
