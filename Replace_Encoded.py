#!/usr/bin/env python3
"""
fast_replace_encoded.py
---------------------------------------------------------------------------
Parallel, high‑speed replacement of “Source” (column C) and “Ethnicity”
(column U) values in a very large Excel workbook, using lookup tables stored
in Replace.xlsx.

Sheet‑to‑column mapping
──────────────────────────────────────────────────────────────────────────────
Replace.xlsx  →  Encoded_Dataset.xlsx
────────────     ─────────────────────
Sheet 1  (Unique_Source)     : column C  (“Source”    or header fallback)
Sheet 2  (Unique_Ethnicity)  : column U  (“Ethnicity” or header fallback)

• In each sheet, **column A** holds the *original* string and **column B**
  holds the *replacement* string.

The script
──────────
1.  Peeks at the first **500 rows** of Encoded_Dataset.xlsx so you can
    confirm the header & structure.
2.  Builds two replacement dictionaries from Replace.xlsx.
3.  Loads the **entire** workbook with **Modin on Ray** (all CPU cores); if
    Modin is absent it falls back to ordinary pandas.
4.  Applies the replacements vectorially (fast, no Python loops).
5.  Saves `Encoded_Dataset_replaced.xlsx`.

Install speed‑up (once):
    pip install "modin[ray]" openpyxl
---------------------------------------------------------------------------"""

import time
from pathlib import Path

# ------------------------------------------------------------------ CONFIG --
DATA_FILE       = Path("Encoded_Dataset.xlsx")
REPLACE_FILE    = Path("Replace.xlsx")
OUTPUT_FILE     = Path("Encoded_Dataset_replaced.xlsx")
PREVIEW_ROWS    = 500
# ---------------------------------------------------------------------------

# ---------- 0. Try Modin for parallelism -----------------------------------
try:
    import modin.pandas as pd
    print("[INFO] Modin detected – running on Ray with all CPU cores.")
except ImportError:
    import pandas as pd
    print("[WARN] Modin not installed – falling back to pandas (single core)."
          "  Install with  `pip install \"modin[ray]\"`  for a big speed‑up.\n")

# ---------- 1. Preview first 500 rows --------------------------------------
print(f"[STEP] Previewing first {PREVIEW_ROWS} rows of {DATA_FILE} …")
preview_df = pd.read_excel(DATA_FILE, nrows=PREVIEW_ROWS, engine="openpyxl")
print(preview_df.head(10))
print(f"[INFO] Workbook preview: {preview_df.shape[1]} columns.\n")

# ---------- 2. Load replacement dictionaries -------------------------------
print(f"[STEP] Loading replacement tables from {REPLACE_FILE} …")
source_map_df = pd.read_excel(REPLACE_FILE, sheet_name=0, usecols="A:B",
                              engine="openpyxl").astype(str).apply(lambda s: s.str.strip())
eth_map_df    = pd.read_excel(REPLACE_FILE, sheet_name=1, usecols="A:B",
                              engine="openpyxl").astype(str).apply(lambda s: s.str.strip())

source_map = dict(zip(source_map_df.iloc[:, 0], source_map_df.iloc[:, 1]))
eth_map    = dict(zip(eth_map_df.iloc[:, 0],   eth_map_df.iloc[:, 1]))

print(f"[INFO] Source replacements loaded:    {len(source_map):>5d}")
print(f"[INFO] Ethnicity replacements loaded: {len(eth_map):>5d}\n")

# ---------- 3. Load full dataset -------------------------------------------
print(f"[STEP] Reading full workbook {DATA_FILE} … (may take a while)")
t0 = time.time()
df = pd.read_excel(DATA_FILE, engine="openpyxl", dtype=str)
print(f"[INFO] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols "
      f"in {time.time() - t0:.1f} s\n")

# ---------- 4. Determine column labels -------------------------------------
# Try by header; if missing, fall back to absolute positions (0‑indexed)
SOURCE_COL_LABEL    = "Source"
ETHNICITY_COL_LABEL = "Ethnicity"

if SOURCE_COL_LABEL not in df.columns:
    # user’s earlier files used “Pt_Ethnicity”; keep a safeguard
    if "source" in df.columns.str.lower():
        SOURCE_COL_LABEL = [c for c in df.columns if c.lower() == "source"][0]
    else:
        SOURCE_COL_LABEL = df.columns[2]   # column C
if ETHNICITY_COL_LABEL not in df.columns:
    if "pt ethnicity" in df.columns.str.lower():
        ETHNICITY_COL_LABEL = [c for c in df.columns
                               if c.lower() == "pt ethnicity"][0]
    else:
        ETHNICITY_COL_LABEL = df.columns[20]  # column U (0‑indexed)

print("[INFO] Using column labels →",
      f"SOURCE: '{SOURCE_COL_LABEL}', ETHNICITY: '{ETHNICITY_COL_LABEL}'\n")

# ---------- 5. Apply replacements (vectorised) -----------------------------
print("[STEP] Replacing values …")
df[SOURCE_COL_LABEL]    = (df[SOURCE_COL_LABEL]
                           .str.strip()
                           .replace(source_map))
df[ETHNICITY_COL_LABEL] = (df[ETHNICITY_COL_LABEL]
                           .str.strip()
                           .replace(eth_map))
print("[INFO] Replacement done.\n")

# ---------- 6. Export -------------------------------------------------------
print(f"[STEP] Saving updated workbook → {OUTPUT_FILE}")
t0 = time.time()
df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"[SUCCESS] File saved in {time.time() - t0:.1f} s")

print("\nAll finished ✔")
