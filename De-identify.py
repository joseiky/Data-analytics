#!/usr/bin/env python3
"""
De-identify MDLNo and Patient-ID across three GIT datasets by assigning
new sequential codes. Generates:

  • MDL_mapping.csv       : Original MDLNo → New MDL code
  • Patient_mapping.csv   : Original Patient-ID → New PAT code
  • Dataset 1. GIT-No-toxins_deidentified.xlsx
  • Dataset 2. CD-Toxins_deidentified.xlsx
  • Dataset 3. Ecoli-Shigella-toxins_deidentified.xlsx

Usage:
  python De-identify.py
"""

import os
import glob
import pandas as pd

# STEP 0: File patterns for the three target workbooks
file_patterns = [
    "Dataset 1. GIT-No-toxins.xlsx",
    "Dataset 2. CD-Toxins.xlsx",
    "Dataset 3. Ecoli-Shigella-toxins.xlsx"
]

# Verify that each pattern matches exactly one file
found_files = {}
for pat in file_patterns:
    matches = glob.glob(pat)
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find any file matching: {pat}")
    if len(matches) > 1:
        raise RuntimeError(f"More than one file matches pattern {pat}: {matches}")
    found_files[pat] = matches[0]

# STEP 1: Scan all sheets to collect unique MDLNo and Patient-ID values
all_mdl = set()
all_pat = set()

print("→ Scanning every sheet of each workbook to collect MDLNo and Patient-ID …")
for friendly_name, filepath in found_files.items():
    xlsx = pd.ExcelFile(filepath, engine="openpyxl")
    for sheet in xlsx.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet, dtype=str)
        if "MDLNo" in df.columns:
            all_mdl.update(df["MDLNo"].dropna().astype(str).unique())
        if "Patient-ID" in df.columns:
            all_pat.update(df["Patient-ID"].dropna().astype(str).unique())

print(f"   • Found {len(all_mdl)} unique MDLNo values")
print(f"   • Found {len(all_pat)} unique Patient-ID values")

# STEP 2: Create mapping dictionaries (orig → new code)
mdl_map = {
    orig: f"MDL{str(i).zfill(5)}"
    for i, orig in enumerate(sorted(all_mdl), start=1)
}
pat_map = {
    orig: f"PAT{str(i).zfill(5)}"
    for i, orig in enumerate(sorted(all_pat), start=1)
}

# STEP 3: Write out the mapping key-files
mdl_key_df = pd.DataFrame({
    "MDLNo_orig": list(mdl_map.keys()),
    "MDLNo_new": list(mdl_map.values())
})
mdl_key_df.to_csv("MDL_mapping.csv", index=False)
print("✓ Wrote MDL_mapping.csv")

pat_key_df = pd.DataFrame({
    "PatientID_orig": list(pat_map.keys()),
    "PatientID_new": list(pat_map.values())
})
pat_key_df.to_csv("Patient_mapping.csv", index=False)
print("✓ Wrote Patient_mapping.csv")

# STEP 4: Apply mappings to every sheet & save de-identified workbooks
for original_pattern, original_path in found_files.items():
    print(f"\n→ De-identifying '{original_pattern}' …")
    xlsx = pd.ExcelFile(original_path, engine="openpyxl")
    output_path = original_path.replace(".xlsx", "_deidentified.xlsx")
    writer = pd.ExcelWriter(output_path, engine="openpyxl")

    for sheet in xlsx.sheet_names:
        df = pd.read_excel(original_path, sheet_name=sheet, dtype=str)
        # Map MDLNo if present
        if "MDLNo" in df.columns:
            df["MDLNo"] = df["MDLNo"].map(lambda v: mdl_map.get(str(v), v))
        # Map Patient-ID if present
        if "Patient-ID" in df.columns:
            df["Patient-ID"] = df["Patient-ID"].map(lambda v: pat_map.get(str(v), v))
        df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"  • Processed sheet '{sheet}' (rows: {len(df)})")

    writer.close()
    print(f"✓ Saved de-identified file: {os.path.basename(output_path)}")

print("\nAll done! Submit the *_deidentified.xlsx files and mapping CSVs.")
