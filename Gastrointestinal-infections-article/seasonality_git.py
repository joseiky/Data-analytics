#!/usr/bin/env python3
"""
Seasonality analysis for GIT No-toxins panel
  • Reads the full sheet into memory (pandas <1.3 fallback)
  • Keeps only Date-Collected, Test, Result columns
  • Counts POSITIVE (Result == 'P') per Test × month
  • Builds a tidy table (CSV) and per-pathogen PNG plots
  • Runs Poisson GLM with harmonic terms (12-month seasonality)
     – Uses a manual Likelihood Ratio Test (LRT) to compute p-value
  • Runs STL decomposition (period = 12)
  • Outputs: 
      ./output/monthly_positive_counts.csv
      ./output/seasonality_summary.csv
      ./output/<Pathogen>_seasonality.png
Author: Your Name
Date: YYYY-MM-DD
"""

import argparse
import pathlib
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy.stats import chi2

# --------------------------- CLI Setup -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--excel",
                    default="Dataset 1. GIT-No-toxins.xlsx",
                    help="Excel workbook name")
parser.add_argument("--sheet", default=0, type=str,
                    help="Sheet name or index")
parser.add_argument("--top", default=None, type=int,
                    help="Analyse N most prevalent pathogens")
parser.add_argument("--target", default=None, type=str,
                    help="Analyse a single pathogen (overrides --top)")
parser.add_argument("--chunksize", default=50000, type=int,
                    help="(Ignored when pandas <1.3—reads whole sheet instead)")
args = parser.parse_args()

# --------------------- 1. Read the Entire Sheet ---------------------
use_cols = ["Date-Collected", "Test", "Result"]
date_counts = defaultdict(lambda: defaultdict(int))  # {Test: {Month: count}}

print("→ Reading entire sheet (no streaming, pandas <1.3)…")
df_full = pd.read_excel(args.excel,
                        sheet_name=args.sheet,
                        usecols=use_cols,
                        engine="openpyxl",
                        dtype={"Result": "category"})

# Keep only positives
df_full = df_full[df_full["Result"] == "P"].copy()

# Convert to Period month
df_full["Month"] = pd.to_datetime(df_full["Date-Collected"],
                                  errors="coerce").dt.to_period("M")

# Group by Test × Month and accumulate counts
grp = df_full.groupby(["Test", "Month"]).size()
for (test, month), n in grp.items():
    date_counts[test][month] += int(n)

print("✓ Finished scanning workbook ({} positive rows)".format(len(df_full)))

# -------------------- 2. Build Tidy DataFrame -----------------------
records = []
for test, months in date_counts.items():
    for month, count in months.items():
        records.append({"Test": test,
                        "Month": month,
                        "Pos_Count": count})
df = pd.DataFrame.from_records(records)
df["Month_start"] = df["Month"].dt.to_timestamp()
df.sort_values(["Test", "Month_start"], inplace=True)

# Choose targets
if args.target:
    targets = [args.target]
elif args.top:
    tot = (df.groupby("Test")["Pos_Count"].sum()
             .sort_values(ascending=False)
             .head(args.top))
    targets = tot.index.tolist()
else:
    targets = df["Test"].unique().tolist()

print(f"→ Analysing {len(targets)} pathogen(s): {targets}")

# Prepare output directory
pathlib.Path("output").mkdir(exist_ok=True)

# Save the full monthly counts for all tests
df.to_csv("output/monthly_positive_counts.csv", index=False)

summary_rows = []

# --------------------- 3. Per-Pathogen Analysis ---------------------
for test in targets:
    # Build a time series indexed by each Month_start
    ser = (df.loc[df["Test"] == test, ["Month_start", "Pos_Count"]]
             .set_index("Month_start")
             .asfreq("MS", fill_value=0)["Pos_Count"])

    # Time index 0,1,2,...,N-1
    months = np.arange(len(ser))

    # Harmonic terms for 12-mo seasonality
    sin12 = np.sin(2 * np.pi * months / 12)
    cos12 = np.cos(2 * np.pi * months / 12)

    # === 3A. Fit the FULL Poisson model (trend + harmonic) ===
    X_full = np.column_stack([months, sin12, cos12])
    X_full = sm.add_constant(X_full, prepend=False)  # shape (n_obs, 4)
    glm_full = sm.GLM(ser.values,
                      X_full,
                      family=sm.families.Poisson()).fit()

    # === 3B. Fit the REDUCED Poisson model (trend only, no harmonic) ===
    X_red = sm.add_constant(months, prepend=False)  # shape (n_obs, 2)
    glm_red = sm.GLM(ser.values,
                     X_red,
                     family=sm.families.Poisson()).fit()

    # === 3C. Compute Likelihood Ratio Test manually ===
    llf_full = glm_full.llf
    llf_red = glm_red.llf
    LR_stat = 2.0 * (llf_full - llf_red)
    p_val = chi2.sf(LR_stat, df=2)  # 2 extra parameters in the full model

    # === 3D. STL decomposition (period=12) ===
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stl = STL(ser, period=12).fit()

    # === 3E. Plot observed + Poisson fit + STL seasonal component ===
    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top panel: Observed vs Poisson fitted
    axes[0].plot(ser.index, ser, marker="o", linewidth=1, label="Observed")
    axes[0].plot(ser.index,
                 glm_full.fittedvalues,
                 linewidth=2, color="C1", label="Poisson fit")
    axes[0].set_title(f"{test}  |  Poisson seasonality LRT p = {p_val:.2e}")
    axes[0].legend()

    # Bottom panel: STL seasonal component
    axes[1].plot(ser.index,
                 stl.seasonal,
                 linewidth=1.5, color="C2", label="STL Seasonal")
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[1].set_title("STL seasonal component (12-mo period)")
    axes[1].legend()

    fig.tight_layout()
    fname = test.replace("/", "_").replace(" ", "_") + "_seasonality.png"
    fig.savefig(f"output/{fname}", dpi=300)
    plt.close(fig)

    # === 3F. Append to summary table ===
    summary_rows.append({
        "Test": test,
        "Total_Pos": int(ser.sum()),
        "LRT_stat": float(LR_stat),
        "Seasonal_LRT_p": float(p_val),
        "STL_Seasonal_Amplitude": float(stl.seasonal.max() - stl.seasonal.min())
    })

# --------------------- 4. Save Summary CSV ---------------------------
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("output/seasonality_summary.csv", index=False)

print("✓ Analysis complete.")
print("  • Monthly counts:      output/monthly_positive_counts.csv")
print("  • Seasonality summary: output/seasonality_summary.csv")
print("  • Individual plots:    in ./output/*.png")
