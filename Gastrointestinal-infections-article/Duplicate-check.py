#!/usr/bin/env python
# clean_git_panel.py  –  Work‑package 0

import pandas as pd, re, pathlib, sys
from datetime import datetime

PATHS = {
    "GIT"   : "GIT-No-toxins.xlsx",
    "CD"    : "CD-Toxins.xlsx",
    "ECOLI" : "Ecoli-Shigella-toxins.xlsx"
}
OUT_CLEAN = "git_panel_clean.pkl"
OUT_DUP   = "duplicates.xlsx"

def snake(col):
    col = col.strip()
    col = re.sub(r"[\s\-]+", "_", col)
    return col.lower()

def load_sheet(path, label):
    df = pd.read_excel(path)
    df.columns = [snake(c) for c in df.columns]

    # dtype fixes
    if "date_collected" in df:
        df["date_collected"] = pd.to_datetime(df["date_collected"], errors="coerce")
        df["month"] = df["date_collected"].dt.to_period("M")

    if "age" in df:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    for cat in ["specimen", "source", "bv_status", "ethnicity", "gender", "test", "result"]:
        if cat in df:
            df[cat] = df[cat].astype("category")

    df["panel"] = label
    return df

def main():
    frames = [load_sheet(p, lbl) for lbl, p in PATHS.items()]
    git_panel = pd.concat(frames, ignore_index=True, sort=False)

    # Duplicate audit (same sample, same test)
    dup_mask = git_panel.duplicated(subset=["mdlno", "test"])
    duplicates = git_panel.loc[dup_mask].copy()

    git_panel.to_pickle(OUT_CLEAN)
    duplicates.to_excel(OUT_DUP, index=False)

    # Console summary
    vc = git_panel.groupby("panel")["mdlno"].nunique()
    tests_per_sample = git_panel.groupby("mdlno")["test"].nunique()
    print("\nUnique MDLN° per panel:\n", vc)
    print("\nTests per sample (five‑number summary):",
          tests_per_sample.describe()[["min","25%","50%","75%","max"]])
    print(f"\nSaved cleaned DataFrame → {OUT_CLEAN}")
    print(f"Saved duplicates audit  → {OUT_DUP}")

if __name__ == "__main__":
    main()
