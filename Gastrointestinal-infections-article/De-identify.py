#!/usr/bin/env python3
"""
De-identify MDLNo and Patient ID across three GIT datasets by assigning
new sequential codes. Generates:

  • MDL_mapping.csv       : Original MDLNo → New MDL code
  • Patient_mapping.csv   : Original Patient ID → New PAT code
  • CD_Toxins_deidentified.xlsx
  • Ecoli_Shigella_deidentified.xlsx
  • GIT_No_Toxins_deidentified.xlsx

Usage:
  python deidentify_datasets.py
"""

import pandas as pd

# ------------------ 1. Define input file paths -------------------
files = {
    "CD_Toxins": "/mnt/data/CD-Toxins.xlsx",
    "Ecoli_Shigella": "/mnt/data/Ecoli-Shigella-toxins.xlsx",
    "GIT_No_Toxins": "/mnt/data/GIT-No-toxins.xlsx"
}

# ------------------ 2. Read each Excel into a pandas DataFrame -------------------
dfs = {}
for shortname, filepath in files.items():
    dfs[shortname] = pd.read_excel(filepath)

# ------------------ 3. Collect all unique MDLNos and Patient IDs -------------------
all_mdl = set()
all_pat = set()

for df in dfs.values():
    # Only accumulate if the column exists
    if "MDLNo" in df.columns:
        all_mdl.update(df["MDLNo"].dropna().astype(str).unique())
    if "Patient ID" in df.columns:
        all_pat.update(df["Patient ID"].dropna().astype(str).unique())

# ------------------ 4. Create mapping dictionaries -------------------
# We will assign codes like MDL00001, MDL00002, … and PAT00001, PAT00002, …
mdl_map = {
    original: f"MDL{str(i).zfill(5)}"
    for i, original in enumerate(sorted(all_mdl), start=1)
}
pat_map = {
    original: f"PAT{str(i).zfill(5)}"
    for i, original in enumerate(sorted(all_pat), start=1)
}

# ------------------ 5. Save mapping key files (for reverse lookup) -------------------
mdl_key_df = pd.DataFrame({
    "MDLNo_orig": list(mdl_map.keys()),
    "MDLNo_new": list(mdl_map.values())
})
mdl_key_df.to_csv("MDL_mapping.csv", index=False)

pat_key_df = pd.DataFrame({
    "PatientID_orig": list(pat_map.keys()),
    "PatientID_new": list(pat_map.values())
})
pat_key_df.to_csv("Patient_mapping.csv", index=False)

# ------------------ 6. Apply mappings to each DataFrame & save -------------------
for shortname, df in dfs.items():
    # Convert columns to string (in case some IDs are numeric)
    if "MDLNo" in df.columns:
        df["MDLNo"] = df["MDLNo"].astype(str).map(mdl_map)
    if "Patient ID" in df.columns:
        df["Patient ID"] = df["Patient ID"].astype(str).map(pat_map)

    outname = f"{shortname}_deidentified.xlsx"
    df.to_excel(outname, index=False)
    print(f"Saved de-identified dataset: {outname}")

print("⇒ Mapping keys saved as MDL_mapping.csv & Patient_mapping.csv")
