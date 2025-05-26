#!/usr/bin/env python3
"""
bulk_replace.py
---------------------------------------------------------------------------
Fast, parallel replacement of “Source” (column C) and “Ethnicity” (column J)
values in a large Excel workbook, using lookup tables stored in
Replacement.xlsx (two sheets).

• Sheet 1 in Replacement.xlsx → replacements for column C (“Source”)
• Sheet 2 in Replacement.xlsx → replacements for column J (“Ethnicity”)

The script

1.  Peeks at the **first 500 rows** of Sorted_Dataset_GIT.xlsx (all columns) so
    you can sanity‑check the header / data types.
2.  Builds two mapping dictionaries from Replacement.xlsx.
3.  Loads the *entire* Sorted_Dataset_GIT.xlsx with **Modin** on Ray (all CPU
    cores).  If Modin is not installed it falls back to ordinary pandas
    (single‑core but still vectorised).
4.  Applies the replacements vectorially (very fast).
5.  Exports an updated Excel file:
        Sorted_Dataset_GIT_replaced.xlsx
---------------------------------------------------------------------------

Usage
$ python3 bulk_replace.py

Dependencies
    pip install "modin[ray]" pandas openpyxl
---------------------------------------------------------------------------"""

import os
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------ CONFIG --
SOURCE_FILE      = Path("Sorted_Dataset_GIT.xlsx")
REPLACEMENT_FILE = Path("Replacement.xlsx")
OUTPUT_FILE      = Path("Sorted_Dataset_GIT_replaced.xlsx")
PREVIEW_ROWS     = 500         # how many rows to peek
# ---------------------------------------------------------------------------

# ---------- 0. Parallel backend (Modin) ------------------------------------
try:
    import modin.pandas as pd             # Modin will spin up Ray, all cores
    print("[INFO] Using Modin + Ray backend (parallel, all CPUs) …")
except ImportError:
    import pandas as pd
    print("[WARN] Modin not found → falling back to pandas (single core).")
    print("       To parallelise,  `pip install \"modin[ray]\"` and re‑run.\n")

# ---------- 1. Preview first 500 rows to verify ----------------------------
print(f"[STEP] Previewing first {PREVIEW_ROWS} rows of {SOURCE_FILE} …")
preview = pd.read_excel(SOURCE_FILE, nrows=PREVIEW_ROWS, engine="openpyxl")
print(preview.head(10))               # print a small glimpse to stdout
print(f"[INFO] Sheet has {preview.shape[1]} columns. Continue …\n")

# ---------- 2. Build replacement dictionaries -----------------------------
print(f"[STEP] Loading replacement tables from {REPLACEMENT_FILE} …")
src_map_df  = pd.read_excel(REPLACEMENT_FILE, sheet_name=0, usecols="A:B",
                            engine="openpyxl").astype(str).apply(lambda s: s.str.strip())
eth_map_df  = pd.read_excel(REPLACEMENT_FILE, sheet_name=1, usecols="A:B",
                            engine="openpyxl").astype(str).apply(lambda s: s.str.strip())

src_map = dict(zip(src_map_df.iloc[:, 0], src_map_df.iloc[:, 1]))
eth_map = dict(zip(eth_map_df.iloc[:, 0], eth_map_df.iloc[:, 1]))

print(f"[INFO] Source replacements loaded:     {len(src_map):>5d} keys")
print(f"[INFO] Ethnicity replacements loaded:  {len(eth_map):>5d} keys\n")

# ---------- 3. Load full dataset -------------------------------------------
print(f"[STEP] Reading full workbook {SOURCE_FILE} … (this may take a minute)")
start = time.time()
df = pd.read_excel(SOURCE_FILE, engine="openpyxl", dtype=str)   # keep everything as str
print(f"[INFO] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols in "
      f"{time.time() - start:.1f} s\n")

# ---------- 4. Vectorised replacement --------------------------------------
# Column C is index 2, but safer to use the header if present.
# Fallback to index if header names differ.
SOURCE_COL_NAME    = "Source"     # adjust if your header differs
ETHNICITY_COL_NAME = "Ethnicity"  # adjust if your header differs

if SOURCE_COL_NAME not in df.columns:
    SOURCE_COL_NAME    = df.columns[2]   # column C fallback (0‑indexed)
if ETHNICITY_COL_NAME not in df.columns:
    ETHNICITY_COL_NAME = df.columns[9]   # column J fallback

print("[STEP] Applying replacements …")
df[SOURCE_COL_NAME]    = df[SOURCE_COL_NAME]   .str.strip().replace(src_map)
df[ETHNICITY_COL_NAME] = df[ETHNICITY_COL_NAME].str.strip().replace(eth_map)
print("[INFO] Replacement complete.\n")

# ---------- 5. Save ---------------------------------------------------------
print(f"[STEP] Writing updated workbook → {OUTPUT_FILE} …")
start = time.time()
df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"[SUCCESS] File saved. ({time.time() - start:.1f} s elapsed)")

# ---------------------------------------------------------------------------
print("\nAll done ✔")
