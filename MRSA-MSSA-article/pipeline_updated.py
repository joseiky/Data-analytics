#!/usr/bin/env python3
"""
pipeline_updated_partA.py

Part A:  
  • Imports, setup, data loading & cleaning  
  • Fig 1 (overview)  
  • Fig 2 (single grouped BV×Test)  
  • Fig 3 (tripanel: AgeBin, Ethnicity, Pathology)  

Modifications included:
  • All panels labeled A, B, C…  
  • Fig 2: single grouped barplot (S. aureus, MRSA, MSSA, AST) with BV-Status proportions  
  • Fig 3: two small panels (AgeBin, Ethnicity) + one large panel (all 9 cytology pathologies)  
    – single BV-Status legend maintained  
  • All `.reindex(..., 0)` calls fixed to `.reindex(..., fill_value=0)`  
  • Export of plot-data for Figs 1–3  
  • “S. aureus” used everywhere  
  • Present/Absent labels instead of 0/1  
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_context("paper")
sns.set(style="whitegrid", palette="Set2"); plt.rcParams.update({"font.size":9})

# ── OUTPUT folder ───────────────────────────────────────────────────────────────
OUT = Path("reanalysed_outputs")
OUT.mkdir(exist_ok=True)

# ── INPUT CSV ───────────────────────────────────────────────────────────────────
CSV_FILE = "Cleaned_Data.csv"

def print_qc(msg):
    print(f"▶ {msg}")

# ── Helper functions ────────────────────────────────────────────────────────────
def chi_or_fisher(tab, min_exp=5):
    """Chi-sq or Fisher’s exact for 2×2 sparse tables."""
    if tab.shape == (2,2):
        chi2, p, dof, exp = chi2_contingency(tab, correction=False)
        if (exp < min_exp).any():
            _, p = fisher_exact(tab)
            return {"stat": np.nan, "p": p, "method": "Fisher"}
    chi2, p, dof, exp = chi2_contingency(tab, correction=False)
    return {"stat": chi2, "p": p, "method": f"Chi2(df={dof})"}

def kruskal_test(df_sub, group_col, value_col, log_transform=True):
    """Return Kruskal–Wallis p-value (with optional log10 transform)."""
    df0 = df_sub[[group_col, value_col]].dropna()
    if df0.empty:
        return {"stat": np.nan, "p": 1.0, "method": "Kruskal"}
    if log_transform:
        df0[value_col] = np.log10(df0[value_col].replace(0, np.nan)).dropna()
    groups = [g[value_col].values for _, g in df0.groupby(group_col)]
    if len(groups) < 2:
        return {"stat": np.nan, "p": 1.0, "method": "Kruskal"}
    stat, p = kruskal(*groups)
    return {"stat": stat, "p": p, "method": "Kruskal"}

def add_pval_annotation(ax, p, xpos=0.5, ypos=0.92):
    """Add stars/ns text based on p-value to a single Axes."""
    if np.isnan(p) or p >= 0.05:
        txt = "ns (p≥0.05)"
    elif p < 0.001:
        txt = "*** p<0.001"
    elif p < 0.01:
        txt = "** p<0.01"
    else:
        txt = "* p<0.05"
    ax.text(xpos, ypos, txt, ha="center", va="center", transform=ax.transAxes)

def savefig(fname):
    plt.tight_layout()
    plt.savefig(OUT/fname, dpi=300)
    plt.close()

# ── LOAD & CLEAN ────────────────────────────────────────────────────────────────
print_qc(f"Loading '{CSV_FILE}' …")
df = pd.read_csv(CSV_FILE, dtype=str)
print_qc(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

df["Date-Collected"] = pd.to_datetime(df["Date-Collected"], errors="coerce")
df = df[df["Date-Collected"].notna()].copy()
df["Month"] = df["Date-Collected"].dt.to_period("M").dt.to_timestamp()


print_qc(f"Rows with valid Date-Collected: {len(df)}")
print_qc("Sample parsed dates:")
print(df["Date-Collected"].head(10).tolist())

# Strip whitespace
for c in df.select_dtypes("object"):
    df[c] = df[c].str.strip()

# Drop 'NT'
df = df[df.get("BV-Status","") != "NT"]

# Convert numeric fields
if "Age" in df: df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
if "Concentration" in df: df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
if "Date-Collected" in df: df["Date-Collected"] = pd.to_datetime(df["Date-Collected"], errors="coerce")

# Flag columns → int
infection_cols = ["TRICH","Actino","Atrophy","CD","BV","Atrophic VAG"]
cyto_cols      = ["NILM","ECA","AGC","ASC-H","ASCUS","RCC","LSIL","HSIL","FQC"]
for col in infection_cols + cyto_cols:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# Create AgeBin
if "Age" in df:
    bins = list(range(0,101,10))
    labels = [f"{i}–{i+9}" for i in bins[:-1]]
    df["AgeBin"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)

# Prepare collectors
results   = []
plot_data = {}

# ==============================================================================
# FIG 1 – OVERVIEW (2×3)
# ==============================================================================
print_qc("Fig 1…")
fig, axs = plt.subplots(2, 3, figsize=(18,10))
sns.despine(left=True)

# Label panels A–F
for idx, ax in enumerate(fig.axes):
    ax.text(0.02, 0.95, chr(65+idx),
            transform=ax.transAxes, fontsize=14, fontweight="bold")

# A: Age dist.
ax = axs[0,0]
if "Age" in df:
    data = df["Age"].dropna().astype(int)
    plot_data["Fig1_Age"] = data.value_counts().sort_index()
    sns.histplot(data, bins=30, ax=ax)
    ax.set(title="Age Distribution", xlabel="Age", ylabel="Count")
else:
    ax.axis("off")

# B: BV-Status dist.
ax = axs[0,1]
bv = df["BV-Status"].value_counts().reindex(["BV-N","BV-P","BV-T"], fill_value=0)
plot_data["Fig1_BV"] = bv
bv.plot.barh(ax=ax)
ax.set(title="BV Status", xlabel="Count")

# C: Specimen type
ax = axs[0,2]
if "Specimen" in df:
    sp = df["Specimen"].value_counts()
    plot_data["Fig1_Specimen"] = sp
    sp.plot.barh(ax=ax)
    ax.set(title="Specimen Type", xlabel="Count")
else:
    ax.axis("off")

# D: Top sources
ax = axs[1,0]
if "Source" in df:
    src = df["Source"].value_counts().head(20)
    plot_data["Fig1_Source"] = src
    src.plot.bar(ax=ax)
    ax.set(title="Top Sources", ylabel="Count")
    ax.tick_params(axis="x", rotation=45)
else:
    ax.axis("off")

# E: Ethnicity
ax = axs[1,1]
if "Ethnicity" in df:
    eth = df["Ethnicity"].value_counts().head(15)
    plot_data["Fig1_Ethnicity"] = eth
    eth.plot.barh(ax=ax)
    ax.set(title="Ethnicity", xlabel="Count")
else:
    ax.axis("off")

# F: blank
axs[1,2].axis("off")

savefig("fig_MAIN1.png")
print_qc("✔ Fig 1\n")

# ==============================================================================
# FIG 2 – SINGLE GROUPED BAR (BV × Tests)
# ==============================================================================
print_qc("Fig 2…")
tests   = ["S. aureus","MRSA","MSSA","AST"]
bv_cats = ["BV-N","BV-P","BV-T"]

# Build count table
tbl2 = pd.DataFrame({
    t: (df[df["Test"]==t]["BV-Status"]
         .value_counts()
         .reindex(bv_cats, fill_value=0))
    for t in tests
}).T
plot_data["Fig2"] = tbl2

# Chi-sq/Fisher
chi = chi_or_fisher(tbl2.values)
chi.update({"test":"BV vs Tests","n":int(tbl2.values.sum())})
results.append(chi)

# Plot
fig, ax = plt.subplots(figsize=(10,6))
tbl2.plot.bar(ax=ax, edgecolor="k")
ax.set(title="BV Status by Test", xlabel="Test", ylabel="Count")
ax.legend(title="BV Status", bbox_to_anchor=(1.02,1))
add_pval_annotation(ax, chi["p"])
ax.text(0.02, 0.95, "A", transform=ax.transAxes,
        fontsize=14, fontweight="bold")

savefig("fig_MAIN2.png")
print_qc("✔ Fig 2\n")

# ==============================================================================
# FIG 3 – TRIPANEL (BV by AgeBin, Ethnicity, Pathology)
# ==============================================================================
print_qc("Fig 3…")
plots = []
titles = []

# (A) AgeBin
if "AgeBin" in df:
    ct = pd.crosstab(df["AgeBin"], df["BV-Status"]).reindex(columns=bv_cats, fill_value=0)
    plots.append(ct); titles.append("BV by Age Bin")
    r = chi_or_fisher(ct.values)
    r.update({"test":"BV vs AgeBin","n":int(ct.values.sum())})
    results.append(r)

# (B) Ethnicity
if "Ethnicity" in df:
    ce = pd.crosstab(df["Ethnicity"], df["BV-Status"]).reindex(columns=bv_cats, fill_value=0)
    plots.append(ce); titles.append("BV by Ethnicity")
    r = chi_or_fisher(ce.values)
    r.update({"test":"BV vs Ethnicity","n":int(ce.values.sum())})
    results.append(r)

# (C) Cytology pathologies (all 9)
tab = pd.DataFrame({
    c: df[df[c]==1]["BV-Status"].value_counts().reindex(bv_cats, fill_value=0)
    for c in cyto_cols
}).T
plots.append(tab); titles.append("BV in Cytology-Positive")
r = chi_or_fisher(tab.values)
r.update({"test":"BV vs Cytology","n":int(tab.values.sum())})
results.append(r)

# Layout
n = len(plots)
cols = 3
rows = int(np.ceil(n/cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*4.5))
axs = axs.flatten()

for i, (tab, title) in enumerate(zip(plots, titles)):
    ax = axs[i]
    tab.plot.bar(stacked=False, ax=ax, edgecolor="k")
    ax.set(title=title, ylabel="Count")
    ax.tick_params(axis="x", rotation=90)
    # p-value
    pval = next((r["p"] for r in results if r["test"].split(" vs ")[1] in title), np.nan)
    add_pval_annotation(ax, pval)
    ax.text(0.02, 0.95, chr(65+i), transform=ax.transAxes,
            fontsize=14, fontweight="bold")

# Turn off extra axes
for j in range(n, len(axs)):
    axs[j].axis("off")

savefig("fig_MAIN3.png")
print_qc("✔ Fig 3\n")

# ==============================================================================
# SECTION D: MAIN FIG 4 – BV & S. aureus vs Infection Pathologies (1×2 canvas)
# ==============================================================================
print_qc("Generating Main Fig 4 (BV & S. aureus vs infections) on one canvas…")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.despine(left=True)

# D1: BV vs each infection pathology
ax = axes[0]
mat = []
labels = []
pvals = []
for ip in infection_cols:
    if ip in df.columns:
        pos = df[df[ip] == 1]
        if pos.empty:
            print_qc(f"⚠️ No '{ip}' positives; skipping D1 {ip}.")
            continue
        counts = pos["BV-Status"].value_counts().reindex(["BV-N","BV-P","BV-T"], fill_value=0)
        mat.append(counts)
        labels.append(ip)

        absent_counts = df[df[ip] == 0]["BV-Status"].value_counts().reindex(["BV-N","BV-P","BV-T"], fill_value=0).values
        present_counts = counts.values
        cont = np.vstack([present_counts, absent_counts])
        res = chi_or_fisher(cont)
        res.update({"test": f"BV vs {ip}", "n": int(cont.sum())})
        results.append(res)
        pvals.append(res["p"])
    else:
        print_qc(f"⚠️ '{ip}' missing; skipping D1 {ip}.")

if mat:
    plot_df = pd.DataFrame(mat, index=labels, columns=["BV-N","BV-P","BV-T"])
    plot_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="k")
    ax.set_title("BV-Status vs Infection Pathologies")
    ax.set_xlabel("Infection Pathology"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="BV-Status", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
    add_pval_annotation(ax, min(pvals))
else:
    ax.text(0.5, 0.5, "No valid infection‐pathology vs BV data", ha="center", va="center")
    ax.set_title("BV vs infections (skipped)")

# D2: S. aureus (+) vs each infection pathology (original logic)
ax = axes[1]
sa_pos = df[(df["Test"] == "S. aureus") & (df["Result"] == "P")]
mat2 = []
labels2 = []
for ip in infection_cols:
    if ip in sa_pos.columns:
        cnt = sa_pos[ip].sum()
        if cnt == 0:
            print_qc(f"⚠️ No S. aureus + '{ip}'; skipping D2 {ip}.")
            continue
        mat2.append(cnt)
        labels2.append(ip)
    else:
        print_qc(f"⚠️ '{ip}' missing in S. aureus subset; skipped D2 {ip}.")

if mat2:
    pd.Series(mat2, index=labels2).plot(kind="bar", ax=ax, color="orchid", edgecolor="k")
    ax.set_title("S. aureus (+) by Infection Pathology")
    ax.set_xlabel("Infection Pathology"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
else:
    ax.text(0.5, 0.5, "No valid S. aureus vs infection data", ha="center", va="center")
    ax.set_title("S. aureus vs infections (skipped)")

savefig("fig_MAIN4_BV_Saureus_vs_infections.png")
print_qc("✔ Completed Main Fig 4.\n")

# ==============================================================================
# SECTION E: MAIN FIG 5 – Composite heatmap (BV, Sa, MRSA vs cytology)
# ==============================================================================
print_qc("Generating Main Fig 5 (composite heatmap)…")
factors = {}

# BV → 1 if "BV-P", 0 if "BV-N"  (BV-T will become NaN here)
if "BV-Status" in df.columns:
    factors["BV"] = df["BV-Status"].map({"BV-P": 1, "BV-N": 0})
else:
    print_qc("⚠️ 'BV-Status' missing; skipping BV in composite heatmap.")

# S. aureus → 1 if Test=="S. aureus" & Result=="P" (else 0, no NaN)
# Give the shorter alias “Sa” so later code (Fig S9) finds it
if "Test" in df.columns and "Result" in df.columns:
    factors["S. aureus"] = ((df["Test"] == "S. aureus") & (df["Result"] == "P")).astype(int)
    factors["Sa"] = factors["S. aureus"]          # ← NEW
else:
    print_qc("⚠️ 'S. aureus' data missing; skipping Sa in composite heatmap.")

# MRSA → 1 if Test=="MRSA", 0 otherwise
if "Test" in df.columns:
    factors["MRSA"] = (df["Test"] == "MRSA").astype(int)
else:
    print_qc("⚠️ 'MRSA' data missing; skipping MRSA in composite heatmap.")

# AgeBin (already string‐encoded; missing = "NA")
if "AgeBin" in df.columns:
    factors["AgeBin"] = df["AgeBin"].astype(str).fillna("NA")
else:
    print_qc("⚠️ 'AgeBin' missing; skipping AgeBin in composite heatmap.")

# Ethnicity (string; missing = "NA")
if "Ethnicity" in df.columns:
    factors["Ethnicity"] = df["Ethnicity"].astype(str).fillna("NA")
else:
    print_qc("⚠️ 'Ethnicity' missing; skipping Ethnicity in composite heatmap.")

path_cols = ["NILM", "ASCUS", "LSIL", "RCC", "ASC-H", "HSIL", "AGC"]
heat = pd.DataFrame(index=factors.keys(), columns=path_cols, dtype=float)

for fac, vec in factors.items():
    for p in path_cols:
        if p not in df.columns:
            continue
        pat_ind = df[p]  # 0/1 numeric

        if fac in ["BV", "Sa", "MRSA"]:
            # ---- DROP any NaN in vec before casting to int ----
            mask = vec.notna()
            if mask.sum() == 0:
                heat.loc[fac, p] = np.nan
                continue

            x = vec[mask].astype(int)
            y = pat_ind[mask]
            tbl = pd.crosstab(x, y)

            if tbl.shape == (2, 2) and (tbl > 0).all().all():
                a = tbl.loc[1, 1]
                b = tbl.loc[1, 0]
                c = tbl.loc[0, 1]
                d = tbl.loc[0, 0]
                orr = (a / (b + 1e-9)) / (c / (d + 1e-9))
                heat.loc[fac, p] = np.log10(orr + 1e-9)
            else:
                heat.loc[fac, p] = np.nan

        else:
            # AgeBin or Ethnicity: vec is a string series ("0–10","NA", etc.)
            tab2 = pd.crosstab(pat_ind, vec)
            if tab2.shape[1] < 2:
                heat.loc[fac, p] = np.nan
            else:
                res = chi_or_fisher(tab2.values)
                res.update({"test": f"{fac} vs {p}", "n": int(tab2.values.sum())})
                results.append(res)
                heat.loc[fac, p] = -np.log10(res["p"] + 1e-12)

# Drop any pathology columns that are all-NaN
heat2 = heat.dropna(how="all", axis=1)

if not heat2.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heat2.astype(float),
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .8}
    )
    ax.set_title("Composite log₁₀(OR) or –log₁₀(p)\n(factors vs cytology outcomes)")
    savefig("fig_MAIN5_composite_heatmap.png")
    print_qc("✔ Completed Main Fig 5.\n")
else:
    print_qc("⚠️ Composite heatmap empty; no valid data for heatmap.\n")

# ==============================================================================
# SECTION F: MAIN FIG 6 – Interaction Effects (1×4 canvas)
#   Pairs: MRSA×BV, Sa×BV, MRSA×AgeBin, Sa×Eth
# ==============================================================================
print_qc("Generating Main Fig 6 (interaction effects) on one canvas…")
pairs = [("MRSA", "BV"), ("S. aureus", "BV"), ("MRSA", "AgeBin"), ("S. aureus", "Ethnicity")]
n_pairs = len(pairs)
ncols = 2
nrows = int(np.ceil(n_pairs / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*7, nrows*5))
axs = axs.flatten()

for idx, (a, b) in enumerate(pairs):
    ax = axs[idx]
    if a not in factors or b not in factors:
        ax.text(0.5, 0.5, f"⚠️ '{a}' or '{b}' missing; skipped {a}×{b}.", ha="center", va="center")
        ax.set_title(f"Interaction {a}×{b} (skipped)")
        continue

    vec_a = factors[a]
    vec_b = factors[b]
    tab = pd.crosstab(vec_a, vec_b)
    total = int(tab.values.sum())
    if total < 20:
        ax.text(0.5, 0.5, f"⚠️ Only {total} total; skipped {a}×{b}.", ha="center", va="center")
        ax.set_title(f"Interaction {a}×{b} (skipped)")
        continue

    res = chi_or_fisher(tab.values)
    res.update({"test": f"{a}×{b}", "n": total})
    results.append(res)
    tab.plot(kind="bar", stacked=False, ax=ax, colormap="Spectral", edgecolor="k")
    ax.set_title(f"Interaction {a} × {b}")
    ax.set_xlabel(a); ax.set_ylabel("Count")
    ax.legend(title=b, bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    add_pval_annotation(ax, res["p"])

for k in range(n_pairs, len(axs)):
    axs[k].axis("off")

savefig("fig_MAIN6_interaction_effects.png")
print_qc("✔ Completed Main Fig 6.\n")

# ==============================================================================
# SECTION G: MAIN FIG 7 – Temporal Trends (2×2 canvas)
#   Panels: BV-Positive, S. aureus (+), MRSA, MSSA
# ==============================================================================
print_qc("Generating Main Fig 7 (monthly trends)…")
from scipy.stats import linregress

# Ensure Month column exists and is clean
df["Month"] = (
    pd.to_datetime(df["Date-Collected"], errors="coerce")
      .dt.to_period("M").dt.to_timestamp()
)

trend_vars = {
    "BV-Positive":        lambda d: (d["BV-Status"] == "BV-P").sum(),
    "S. aureus (+)":      lambda d: ((d["Test"] == "S. aureus") & (d["Result"] == "P")).sum(),
    "MRSA":               lambda d: (d["Test"] == "MRSA").sum(),
    "MSSA":               lambda d: (d["Test"] == "MSSA").sum(),
}

if df["Month"].notna().any():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (title, fn) in enumerate(trend_vars.items()):
        ax = axes[i]
        y = df.groupby("Month").apply(fn).sort_index()
        x = np.arange(len(y))  # for regression

        y.plot(kind="bar", ax=ax, edgecolor="k", color=sns.color_palette("Set2")[i])
        # Custom x-labels: YYYY-MM only
        labels = [dt.strftime("%Y-%m") for dt in y.index]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set(title=title, xlabel="", ylabel="Count")
        ax.text(0.02, 0.92, chr(65+i), transform=ax.transAxes,
                fontsize=14, fontweight="bold")

        # Linear trend stats
        slope, intercept, r_value, p_value, std_err = linregress(x, y.values)
        if np.isnan(p_value):
            stattxt = "ns (trend)"
        elif p_value < 0.001:
            stattxt = "*** p<0.001 (trend)"
        elif p_value < 0.01:
            stattxt = "** p<0.01 (trend)"
        elif p_value < 0.05:
            stattxt = "* p<0.05 (trend)"
        else:
            stattxt = "ns (p≥0.05 trend)"
        ax.text(0.85, 0.85, stattxt, ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="crimson")

    # Adjust spacing as needed for more vertical room between grids
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.30)  # ← Increase for more vertical space (try 0.35 or 0.40 if needed)

    savefig("fig_MAIN7_trends.png")
    print_qc("✔ Completed Main Fig 7.\n")
else:
    print_qc("⚠️ No valid ‘Date-Collected’; skipped Main Fig 7.\n")
# ==============================================================================
# SECTION H: MAIN FIG 8 – ML ROC & Feature Importances (2×3 grid, no overlaps)
# ==============================================================================
import matplotlib.gridspec as gridspec

print_qc("Generating Main Fig 8 (machine learning)…")

# Prepare data
ml_df = df.dropna(subset=["BV-Status", "Age"]).copy()
ml_df["BV_bin"] = (ml_df["BV-Status"] == "BV-P").astype(int)

X_cont = ml_df[["Age"]]
cat_cols = [c for c in ["Ethnicity","Specimen","Source"] if c in ml_df.columns]
if cat_cols:
    X = pd.concat([X_cont, pd.get_dummies(ml_df[cat_cols], drop_first=True)], axis=1)
else:
    X = X_cont
y = ml_df["BV_bin"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree":       DecisionTreeClassifier(),
    "Random Forest":       RandomForestClassifier(n_estimators=100),
    "XGBoost":             XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

importances = {}             # ← NEW: collect top-10 feature lists
roc_summary = []             # ← NEW: collect mean AUC ± SD


# Create figure + GridSpec
gs  = gridspec.GridSpec(2, 3,
                        height_ratios=[1,1],
                        hspace=0.2,  # reduce vertical gap
                        wspace=0.3)  # keep horizontal gap

fig = plt.figure(figsize=(18,10))  # reduce top margin
# Top: ROC spanning all 3 columns
ax_roc = fig.add_subplot(gs[0, :])
mean_fpr = np.linspace(0,1,100)
for name, model in models.items():
    tprs, aucs = [], []
    for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:,1]
        fpr, tpr, _ = roc_curve(y.iloc[test_idx], probs)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        aucs.append(roc_auc_score(y.iloc[test_idx], probs))
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1]=1.0
    ax_roc.plot(mean_fpr, mean_tpr,
                label=f"{name} (AUC {np.mean(aucs):.2f}±{np.std(aucs):.2f})", lw=2)
ax_roc.plot([0,1],[0,1], "--", color="gray")
ax_roc.set(title="ROC Curves for BV Prediction",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
ax_roc.legend(loc="lower right", fontsize=9)

# Bottom: three Feature Importance subplots
feature_models = ["Logistic Regression","Random Forest","XGBoost"]
for idx, mname in enumerate(feature_models):
    ax = fig.add_subplot(gs[1, idx])
    if mname in models:
        mdl = models[mname]
        mdl.fit(X, y)
        if mname=="Logistic Regression":
            fi = pd.Series(mdl.coef_[0], index=X.columns).abs().sort_values(ascending=False)
        else:
            fi = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
        importances[mname] = fi.head(10)                           # ← NEW
        fi.head(10).plot.bar(ax=ax, edgecolor="k")
        ax.set(title=f"{mname} Top 10 Features", xlabel="Feature", ylabel="Importance")
        ax.tick_params(axis="x", rotation=90, labelsize=9)
        roc_summary.append(                                         # ← NEW
            {"Model": mname, "MeanAUC": np.mean(aucs), "SDAUC": np.std(aucs)}
        )

    else:
        ax.text(0.5,0.5,f"{mname} missing", ha="center", va="center")
        ax.set_title(mname)

# extra breathing room between ROC (top) and 1×3 grid (bottom)
fig.subplots_adjust(top=0.93, bottom=0.22, left=0.07, right=0.98, hspace=1.2, wspace=0.33)

# if any legend still overlaps, shove it just below the ROC axes:
# ax_lower.legend(loc="upper center",
#                 bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)


savefig("fig_MAIN8_ML_ROC_and_FeatureImportances.png")
print_qc("✔ Completed Main Fig 8.\n")

# ==============================================================================
# SECTION S1: SUPP FIG S1 – Descriptive Summary (2×3 canvas)
# ==============================================================================
print_qc("Generating Supplementary Fig S1 (summary counts + log₁₀ conc) on one canvas…")
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
sns.despine(left=True)

# S1a: Cytology flag count (NILM)
ax = axs[0,0]
if "NILM" in df.columns:
    cnts = df["NILM"].value_counts().reindex([0,1], fill_value=0)
    cnts.plot(kind="bar", ax=ax, color="navy", edgecolor="k")
    ax.set_title("Cytology: NILM (0=absent,1=present)")
    ax.set_xlabel("Indicator"); ax.set_ylabel("Count")
else:
    ax.text(0.5, 0.5, "No 'NILM'", ha="center", va="center")
    ax.set_title("NILM (skipped)")

# S1b: Infection flag count (TRICH)
ax = axs[0,1]
if "TRICH" in df.columns:
    cnts = df["TRICH"].value_counts().reindex([0,1], fill_value=0)
    cnts.plot(kind="bar", ax=ax, color="darkred", edgecolor="k")
    ax.set_title("Infection: TRICH (0=absent,1=present)")
    ax.set_xlabel("Indicator"); ax.set_ylabel("Count")
else:
    ax.text(0.5, 0.5, "No 'TRICH'", ha="center", va="center")
    ax.set_title("TRICH (skipped)")

# S1c: S. aureus positive vs negative
ax = axs[0,2]
if "Test" in df.columns and "Result" in df.columns:
    sa_tab = df["Result"][df["Test"] == "S. aureus"].value_counts().reindex(["P","N"], fill_value=0)
    sa_tab.plot(kind="bar", ax=ax, color="darkgreen", edgecolor="k")
    ax.set_title("S. aureus: Positive vs Negative")
    ax.set_xlabel("Result"); ax.set_ylabel("Count")
else:
    ax.text(0.5, 0.5, "No S. aureus data", ha="center", va="center")
    ax.set_title("S. aureus (skipped)")

# S1d: log₁₀ concentration histogram (S. aureus positives)
ax = axs[1,0]
if all(col in df.columns for col in ["Test", "Result", "Concentration"]):
    sa_pos = df[(df["Test"] == "S. aureus") & (df["Result"] == "P")]
    conc = sa_pos["Concentration"].dropna()
    if not conc.empty:
        conc_pos = conc[conc > 0]
        if not conc_pos.empty:
            sns.histplot(np.log10(conc_pos), kde=True, ax=ax, color="teal")
            ax.set_title("log₁₀ S. aureus Concentration")
            ax.set_xlabel("log₁₀(Concentration)"); ax.set_ylabel("Frequency")
        else:
            ax.text(0.5, 0.5, "No positive concentration", ha="center", va="center")
            ax.set_title("log₁₀ conc (skipped)")
    else:
        ax.text(0.5, 0.5, "No concentration data", ha="center", va="center")
        ax.set_title("log₁₀ conc (skipped)")
else:
    ax.text(0.5, 0.5, "Missing columns", ha="center", va="center")
    ax.set_title("log₁₀ conc (skipped)")

# S1e: Age distribution (KDE)
ax = axs[1,1]
if "Age" in df.columns:
    sns.histplot(df["Age"].dropna(), kde=True, ax=ax, color="purple")
    ax.set_title("Age Distribution (supplement)")
    ax.set_xlabel("Age"); ax.set_ylabel("Frequency")
else:
    ax.text(0.5, 0.5, "No Age data", ha="center", va="center")
    ax.set_title("Age (skipped)")

# S1f: blank placeholder
axs[1,2].axis("off")

savefig("fig_SUPP_S1_summary_counts.png")
print_qc("✔ Completed Supplementary Fig S1.\n")

# ==============================================================================
# SECTION S2: SUPP FIG S2 – S. aureus positives vs {Age, Eth, BV, Infections} (2×2)
# ==============================================================================
print_qc("Generating Supplementary Fig S2 (S. aureus vs demographics/infections) in a 2×2 grid…")
sa_pos = df[(df["Test"] == "S. aureus") & (df["Result"] == "P")]

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
sns.despine(left=True)

# S2a: by Age
ax = axs[0,0]
if "Age" in sa_pos.columns:
    age_counts = sa_pos["Age"].dropna().astype(int).value_counts().sort_index()
    age_counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="k")
    ax.set_title("S. aureus (+) Count by Age")
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
else:
    ax.text(0.5, 0.5, "No Age", ha="center", va="center")
    ax.set_title("S. aureus vs Age (skipped)")

# S2b: by Ethnicity
ax = axs[0,1]
if "Ethnicity" in sa_pos.columns:
    tbl = sa_pos["Ethnicity"].value_counts().head(15)
    tbl.plot(kind="bar", ax=ax, color="coral", edgecolor="k")
    ax.set_title("S. aureus (+) by Ethnicity")
    ax.set_xlabel("Ethnicity"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
else:
    ax.text(0.5, 0.5, "No Ethnicity", ha="center", va="center")
    ax.set_title("S. aureus vs Ethnicity (skipped)")

# S2c: by BV-Status
ax = axs[1,0]
if "BV-Status" in sa_pos.columns:
    tbl = sa_pos["BV-Status"].value_counts().reindex(["BV-N","BV-P","BV-T"], fill_value=0)
    tbl.plot(kind="bar", ax=ax, color="seagreen", edgecolor="k")
    ax.set_title("S. aureus (+) by BV-Status")
    ax.set_xlabel("BV-Status"); ax.set_ylabel("Count")
else:
    ax.text(0.5, 0.5, "No BV-Status", ha="center", va="center")
    ax.set_title("S. aureus vs BV (skipped)")

# S2d: S. aureus vs Infections
ax = axs[1,1]
infection_counts = {}
for ip in infection_cols:
    if ip in sa_pos.columns:
        cnt = sa_pos[ip].sum()
        infection_counts[ip] = cnt
    else:
        print_qc(f"⚠️ '{ip}' missing; skipping in S2d.")

if infection_counts:
    pd.Series(infection_counts).plot(kind="bar", ax=ax, color="plum", edgecolor="k")
    ax.set_title("S. aureus (+) Co-occurrence with Infections")
    ax.set_xlabel("Infection Pathology"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
else:
    ax.text(0.5, 0.5, "No infection data", ha="center", va="center")
    ax.set_title("S. aureus vs Infections (skipped)")

savefig("fig_SUPP_S2_Saureus_vs_demo_infections.png")
print_qc("✔ Completed Supplementary Fig S2.\n")

# ==============================================================================
# SECTION S3a: SUPP FIG S3a – AST vs {Age, Ethnicity, BV-Status} (1×3)
# ==============================================================================
print_qc("Generating Supplementary Fig S3a (AST vs Age/Ethnicity/BV-Status) …")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
sns.despine(left=True)

if "Test" in df.columns and "Result" in df.columns and "AST" in df["Test"].unique():
    ast_df = df[df["Test"] == "AST"].copy()
    ast_df["Suscept"] = ast_df["Result"].replace({"AI": "Non-S", "AR": "Non-S", "AS": "S"})
    for idx, col in enumerate(["Age", "Ethnicity", "BV-Status"]):
        ax = axs[idx]
        if col in ast_df.columns:
            if col == "Age":
                tbl = pd.crosstab(ast_df[col].dropna().astype(int), ast_df["Suscept"])
            else:
                tbl = pd.crosstab(ast_df[col], ast_df["Suscept"])
            if tbl.shape[1] < 2:
                ax.text(0.5, 0.5, f"⚠️ Not enough Suscept categories for AST vs {col}; skipped.", ha="center", va="center")
                ax.set_title(f"AST vs {col} (skipped)")
                continue
            res = chi_or_fisher(tbl.values)
            res.update({"test": f"AST vs {col}", "n": int(tbl.values.sum())})
            results.append(res)
            tbl.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="k")
            ax.set_title(f"AST Susceptibility by {col}")
            ax.set_xlabel(col); ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            add_pval_annotation(ax, res["p"])
        else:
            ax.text(0.5, 0.5, f"⚠️ '{col}' missing in AST; skipped.", ha="center", va="center")
            ax.set_title(f"AST vs {col} (skipped)")
else:
    for ax in axs:
        ax.text(0.5, 0.5, "⚠️ AST data missing; skipped S3a.", ha="center", va="center")
        ax.set_title("AST (skipped)")

savefig("fig_SUPP_S3_AST_vs_Age_Eth_BV.png")
print_qc("✔ Completed Supplementary Fig S3a.\n")

# ==============================================================================
# SECTION S3b: SUPP FIG S3b – MRSA & MSSA vs {Age, Ethnicity, BV-Status} (2×3)
# ==============================================================================
print_qc("Generating Supplementary Fig S3b (MRSA/MSSA vs Age/Ethnicity/BV-Status) …")
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
sns.despine(left=True)

if "Test" in df.columns:
    for row_idx, (label, test_val, color) in enumerate([("MRSA", "MRSA", "gold"), ("MSSA", "MSSA", "orchid")]):
        sub = df[df["Test"] == test_val]
        for col_idx, col in enumerate(["Age", "Ethnicity", "BV-Status"]):
            ax = axs[row_idx, col_idx]
            if col in sub.columns:
                if col == "Age":
                    age_counts = sub[col].dropna().astype(int).value_counts().sort_index()
                    age_counts.plot(kind="bar", ax=ax, color=color, edgecolor="k")
                    ax.set_title(f"{label} Count by Age")
                    ax.set_xlabel("Age"); ax.set_ylabel("Count")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
                else:
                    sub[col].value_counts().plot(kind="bar", ax=ax, color=color, edgecolor="k")
                    ax.set_title(f"{label} by {col}")
                    ax.set_xlabel(col); ax.set_ylabel("Count")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.text(0.5, 0.5, f"⚠️ '{col}' missing in {label}; skipped.", ha="center", va="center")
                ax.set_title(f"{label} by {col} (skipped)")
else:
    for row in axs:
        for ax in row:
            ax.text(0.5, 0.5, "⚠️ MRSA/MSSA data missing; skipped S3b.", ha="center", va="center")
            ax.set_title("MRSA/MSSA (skipped)")

savefig("fig_SUPP_S3_MRSA_MSSA_vs_Age_Eth_BV.png")
print_qc("✔ Completed Supplementary Fig S3b.\n")

# ==============================================================================
# SECTION S4: SUPP FIG S4 – Cytology Volcano Plot (single canvas)
# ==============================================================================
print_qc("Generating Supplementary Fig S4 (cytology volcano)…")
volcano_data = []
if "BV-Status" in df.columns:
    for c in cyto_cols:
        if c in df.columns:
            ind = df[c]
            if ind.sum() == 0:
                print_qc(f"⚠️ No positive '{c}'; skipped S4 entry.")
                continue
            tab = pd.crosstab(ind, df["BV-Status"])
            if not all(x in tab.columns for x in ["BV-N", "BV-P"]):
                print_qc(f"⚠️ '{c}' × BV-Status not 2×2; skipped S4 entry.")
                continue
            a = tab.loc[1, "BV-P"] if (1 in tab.index and "BV-P" in tab.columns) else 0
            b = tab.loc[0, "BV-P"] if (0 in tab.index and "BV-P" in tab.columns) else 0
            c_ = tab.loc[1, "BV-N"] if (1 in tab.index and "BV-N" in tab.columns) else 0
            d = tab.loc[0, "BV-N"] if (0 in tab.index and "BV-N" in tab.columns) else 0
            tb_2x2 = np.array([[a, c_], [b, d]])
            chi2_stat, p_chi, dof, exp = chi2_contingency(tb_2x2, correction=False)
            if (exp < 5).any():
                _, p = fisher_exact(tb_2x2)
            else:
                p = p_chi
            orr = (a / (b + 1e-9)) / (c_ / (d + 1e-9)) if (a*b*c_*d) > 0 else np.nan
            volcano_data.append({"Path": c, "logOR": np.log10(orr + 1e-9), "p_raw": p})
        else:
            print_qc(f"⚠️ '{c}' missing; skipped S4 entry.")

    vol_df = pd.DataFrame(volcano_data)
    if not vol_df.empty:
        padj = multipletests(vol_df["p_raw"], method="fdr_bh")[1]
        vol_df["p_adj"] = padj
        vol_df["neglog10p"] = -np.log10(vol_df["p_adj"] + 1e-12)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            data=vol_df, x="logOR", y="neglog10p", hue="p_adj",
            palette="viridis_r", size="neglog10p", sizes=(50,250), legend=False, ax=ax
        )
        ax.axvline(0, ls="--", color="gray")
        ax.set_xlabel("log₁₀(Odds Ratio)"); ax.set_ylabel("-log₁₀(FDR-adj p)")
        ax.set_title("Volcano Plot: Cytology vs BV")
        for _, row in vol_df.iterrows():
            ax.annotate(
                row["Path"],
                xy=(row["logOR"], row["neglog10p"]),
                xytext=(0, -6),         # move label below
                textcoords="offset points",
                ha="center", va="top",
                fontsize=8
            )
        savefig("fig_SUPP_S4_cytology_volcano.png")    
        print_qc("✔ Completed Supplementary Fig S4.\n")
    else:
        print_qc("⚠️ No valid volcano points; skipped S4.\n")
else:
    print_qc("⚠️ 'BV-Status' missing; skipped S4.\n")

# ==============================================================================
# SECTION S5: SUPP FIG S5 – S. aureus concentration vs {BV, AgeBin, each cytology}
#   (3 columns → roughly 3×4 grid)
# ==============================================================================
print_qc("Generating Supplementary Fig S5 (S. aureus concentration) on one canvas…")
if all(col in df.columns for col in ["Test", "Result", "Concentration"]):
    conc_df = df[(df["Test"] == "S. aureus") & (df["Result"] == "P")].copy()
    conc_df = conc_df[conc_df["Concentration"].notna()]
    if not conc_df.empty:
        conc_df["log10Conc"] = np.log10(conc_df["Concentration"].replace(0, np.nan))
        total_plots = 2 + len(cyto_cols)
        ncols = 3
        nrows = int(np.ceil(total_plots / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*7, nrows*5))
        axs = axs.flatten()

        # S5a: log₁₀ conc by BV-Status
        ax = axs[0]
        if "BV-Status" in conc_df.columns:
            res = kruskal_test(conc_df, "BV-Status", "Concentration", log_transform=True)
            sns.boxplot(x="BV-Status", y="log10Conc", data=conc_df, palette="pastel", ax=ax)
            ax.set_title("log₁₀(S. aureus Conc) by BV-Status")
            ax.set_xlabel("BV-Status"); ax.set_ylabel("log₁₀(Concentration)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            add_pval_annotation(ax, res["p"])
        else:
            ax.text(0.5, 0.5, "⚠️ 'BV-Status' missing; skipped S5a.", ha="center", va="center")
            ax.set_title("S5a (skipped)")

        # S5b: log₁₀ conc by AgeBin
        ax = axs[1]
        if "AgeBin" in conc_df.columns:
            res = kruskal_test(conc_df, "AgeBin", "Concentration", log_transform=True)
            sns.boxplot(x="AgeBin", y="log10Conc", data=conc_df, palette="muted", ax=ax)
            ax.set_title("log₁₀(S. aureus Conc) by Age Bin")
            ax.set_xlabel("Age Bin"); ax.set_ylabel("log₁₀(Concentration)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            add_pval_annotation(ax, res["p"])
        else:
            ax.text(0.5, 0.5, "⚠️ 'AgeBin' missing; skipped S5b.", ha="center", va="center")
            ax.set_title("S5b (skipped)")

        # S5c: log₁₀ conc by each cytology
        for idx, path in enumerate(cyto_cols, start=2):
            ax = axs[idx]
            if path in conc_df.columns:
                ind = conc_df[path]
                if ind.sum() == 0 or (ind == 0).sum() == 0:
                    ax.text(0.5, 0.5, f"⚠️ Not enough groups for '{path}' in S5c; skipped.", ha="center", va="center")
                    ax.set_title(f"S5c {path} (skipped)")
                    continue
                pos = conc_df[ind == 1]["log10Conc"].dropna()
                neg = conc_df[ind == 0]["log10Conc"].dropna()
                stat, pval = kruskal(pos, neg)
                plot_df = pd.DataFrame({"Absent": neg, "Present": pos})
                sns.boxplot(data=plot_df, palette="viridis", ax=ax)
                ax.set_title(f"log₁₀ Conc by {path}")
                ax.set_xlabel(path); ax.set_ylabel("log₁₀(Concentration)")
                add_pval_annotation(ax, pval)
                ax.set_xticklabels(["Absent", "Present"], rotation=45, ha="right")
            else:
                ax.text(0.5, 0.5, f"⚠️ '{path}' missing; skipped S5c.", ha="center", va="center")
                ax.set_title(f"S5c {path} (skipped)")

        for k in range(total_plots, len(axs)):
            axs[k].axis("off")

        savefig("fig_SUPP_S5_conc_vs_BV_Age_Pathology.png")
        print_qc("✔ Completed Supplementary Fig S5.\n")
    else:
        print_qc("⚠️ No S. aureus P concentration data; skipped S5.\n")
else:
    print_qc("⚠️ Required columns missing; skipped S5.\n")

# ==============================================================================
# SECTION S6: SUPP FIG S6 – S. aureus Ct‐score vs BV/Age/Pathology (3×4 grid)
# ==============================================================================
print_qc("Generating Supplementary Fig S6 (S. aureus Ct-score) on one canvas…")
if all(col in df.columns for col in ["Test", "Result", "Ct -score"]):
    ct_df = df[(df["Test"] == "S. aureus") & (df["Result"] == "P")].copy()
    ct_df = ct_df[ct_df["Ct -score"].notna()]
    ct_df['Ct -score'] = pd.to_numeric(ct_df['Ct -score'], errors='coerce')
    ct_df['Ct_bin'] = pd.cut(ct_df['Ct -score'], bins=[0,20,30,40], labels=['Low','Medium','High'])
    ct_df['Ct_bin'].value_counts().reindex(['Low','Medium','High']).plot.bar(ax=ax)
    if not ct_df.empty:
        total_plots = 2 + len(cyto_cols)
        ncols = 3
        nrows = int(np.ceil(total_plots / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*7, nrows*5))
        axs = axs.flatten()

        # S6a: Ct vs BV-Status
        # S6a: Dual plot – Boxplot and Binned Barplot for Ct vs BV-Status
        axs_s6a = [axs[0], axs[1]]  # Left and right subplots in your grid

        if "BV-Status" in ct_df.columns:
            # --- Boxplot ---
            sns.boxplot(x="BV-Status", y="Ct -score", data=ct_df, palette="pastel", ax=axs_s6a[0])
            axs_s6a[0].set_title("Boxplot: Ct Scores by BV-Status")
            axs_s6a[0].set_xlabel("BV-Status")
            axs_s6a[0].set_ylabel("Ct -score")
            axs_s6a[0].set_xticklabels(axs_s6a[0].get_xticklabels(), rotation=90, ha="right")

            # --- Binned Barplot ---
            ct_df['Ct_bin'] = pd.cut(ct_df['Ct -score'], bins=[0,20,30,40], labels=['Low','Medium','High'])
            ct_bin_counts = ct_df.groupby("BV-Status")['Ct_bin'].value_counts().unstack().reindex(columns=['Low','Medium','High'])
            ct_bin_counts.plot.bar(ax=axs_s6a[1], color=["#8dd3c7", "#ffffb3", "#bebada"], edgecolor="k")
            axs_s6a[1].set_title("Binned Ct Scores by BV-Status")
            axs_s6a[1].set_xlabel("BV-Status")
            axs_s6a[1].set_ylabel("Count")
            axs_s6a[1].set_xticklabels(axs_s6a[1].get_xticklabels(), rotation=90, ha="right")
        else:
            axs_s6a[0].text(0.5, 0.5, "BV-Status missing; skipped S6a.", ha="center", va="center")
            axs_s6a[0].set_title("S6a (skipped)")
            axs_s6a[1].axis("off")  # Hide second subplot if missing 

        # S6b: Ct vs AgeBin
        ax = axs[1]
        if "AgeBin" in ct_df.columns:
            res = kruskal_test(ct_df, "AgeBin", "Ct -score", log_transform=False)
            sns.boxplot(x="AgeBin", y="Ct -score", data=ct_df, palette="muted", ax=ax)
            ax.set_title("Ct-score by Age Bin")
            ax.set_xlabel("Age Bin"); ax.set_ylabel("Ct -score")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            add_pval_annotation(ax, res["p"])
        else:
            ax.text(0.5, 0.5, "⚠️ 'AgeBin' missing; skipped S6b.", ha="center", va="center")
            ax.set_title("S6b (skipped)")

        # S6c: Ct vs each cytology
        for idx, path in enumerate(cyto_cols, start=2):
            ax = axs[idx]
            if path in ct_df.columns:
                ind = ct_df[path]
                if ind.sum() == 0 or (ind == 0).sum() == 0:
                    ax.text(0.5, 0.5, f"⚠️ Not enough groups for '{path}' in S6c; skipped.", ha="center", va="center")
                    ax.set_title(f"S6c {path} (skipped)")
                    continue
                pos = ct_df[ind == 1]["Ct -score"].dropna()
                neg = ct_df[ind == 0]["Ct -score"].dropna()
                stat, pval = kruskal(pos, neg)
                plot_df = pd.DataFrame({"Absent": neg, "Present": pos})
                sns.boxplot(data=plot_df, palette="viridis", ax=ax)
                ax.set_title(f"Ct-score by {path}")
                ax.set_xlabel(path); ax.set_ylabel("Ct -score")
                add_pval_annotation(ax, pval)
                ax.set_xticklabels(["Absent", "Present"], rotation=45, ha="right")
            else:
                ax.text(0.5, 0.5, f"⚠️ '{path}' missing; skipped S6c.", ha="center", va="center")
                ax.set_title(f"S6c {path} (skipped)")

        for k in range(total_plots, len(axs)):
            axs[k].axis("off")

        savefig("fig_SUPP_S6_Ct_vs_BV_Age_Pathology.png")
        print_qc("✔ Completed Supplementary Fig S6.\n")
    else:
        print_qc("⚠️ No S. aureus Ct-score data; skipped S6.\n")
else:
    print_qc("⚠️ Required columns missing; skipped S6.\n")

# ==============================================================================
# SECTION S7: SUPP FIG S7 – S. aureus Concentration vs Cytology (stripplot, 3×3 or 3×4)
# ==============================================================================
print_qc("Generating Supplementary Fig S7 (conc vs cytology outcomes) on one canvas…")
conc_df2 = pd.DataFrame()        # ← new stub so the name always exists
if all(col in df.columns for col in ["Test", "Result", "Concentration"]):
    conc_df2 = df[(df["Test"] == "S. aureus") & (df["Result"] == "P")].copy()
    conc_df2 = conc_df2[conc_df2["Concentration"].notna()]
    conc_df2['Concentration'] = pd.to_numeric(conc_df2['Concentration'], errors='coerce')
    conc_df2['log10Conc'] = np.log10(conc_df2['Concentration'].replace(0, np.nan))

if not conc_df2.empty and cyto_cols:
    total_plots = len(cyto_cols)
    figS7, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=False, sharey=False)
    axs = axs.flatten()          # ← critical: gives a 1-D list of 9 axes
   
    for idx, path in enumerate(cyto_cols):
        ax = axs[idx] if total_plots > 1 else axs[0]
        if path in conc_df2.columns:
            ind = conc_df2[path]
            # If not enough data for both groups, skip this plot
            if ind.sum() == 0 or (ind == 0).sum() == 0:
                ax.text(0.5, 0.5, f"⚠️ Not enough groups for '{path}' in S7; skipped.", ha="center", va="center")
                ax.set_title(f"S7 {path} (skipped)")
                continue
            # Kruskal-Wallis p-value
            stat, pval = kruskal(
                conc_df2[ind == 1]["log10Conc"].dropna(),
                conc_df2[ind == 0]["log10Conc"].dropna()
            )
            # Stripplot
            sns.stripplot(
                x=ind.map({0: "Absent", 1: "Present"}),
                y="log10Conc",
                data=conc_df2.assign(**{path: ind}),
                jitter=0.3,
                alpha=0.6,
                ax=ax,
                palette="Paired"
            )
            ax.set_title(f"log₁₀ Conc by {path}")
            ax.text(0.02, 0.95, chr(65+idx), transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.set_xlabel(path)
            ax.set_ylabel("log₁₀(Concentration)")
            add_pval_annotation(ax, pval)
            ax.set_xticklabels(["Absent", "Present"], rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, f"⚠️ '{path}' missing; skipped S7.", ha="center", va="center")
            ax.set_title(f"S7 {path} (skipped)")

    # Turn off any extra axes if you have more subplots than cytology outcomes
    if total_plots < len(axs):
        for k in range(total_plots, len(axs)):
            axs[k].axis("off")

    figS7.tight_layout()

    savefig("fig_SUPP_S7_conc_vs_cytology_strip.png")
    print_qc("✔ Completed Supplementary Fig S7.\n")
else:
    print_qc("⚠️ S7 skipped: no data or no cytology columns.\n")

# ==============================================================================
# SECTION S8: SUPP FIG S8 – Outcomes vs Source/Specimen (2×2 canvas)
# ==============================================================================
print_qc("Generating Supplementary Fig S8 (outcomes vs source/specimen) on one canvas…")
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
sns.despine(left=True)

# S8a: BV-Status vs Top 12 Sources
ax = axs[0,0]
if "BV-Status" in df.columns and "Source" in df.columns:
    top_src = df["Source"].value_counts().head(12).index
    sub = df[df["Source"].isin(top_src)]
    tbl = pd.crosstab(sub["Source"], sub["BV-Status"]).reindex(columns=["BV-N","BV-P","BV-T"], fill_value=0)
    res = chi_or_fisher(tbl.values); res.update({"test": "BV vs Source", "n": int(tbl.values.sum())})
    results.append(res)
    tbl.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="k")
    ax.set_title("BV Status by Top 12 Sources")
    ax.set_xlabel("Source"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="BV-Status", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
    add_pval_annotation(ax, res["p"])
else:
    ax.text(0.5, 0.5, "⚠️ 'BV-Status' or 'Source' missing; skipped S8a.", ha="center", va="center")
    ax.set_title("S8a (skipped)")

# S8b: S. aureus vs Top 8 Sources
ax = axs[0,1]
if "Test" in df.columns and "Result" in df.columns and "Source" in df.columns:
    sa_sub = df[df["Test"] == "S. aureus"]
    top_src = sa_sub["Source"].value_counts().head(8).index
    sa_sub2 = sa_sub[sa_sub["Source"].isin(top_src)]
    sa_sub2["sa_pos"] = (sa_sub2["Result"] == "P").astype(int)
    tbl = pd.crosstab(sa_sub2["Source"], sa_sub2["sa_pos"]).reindex(columns=[0,1], fill_value=0)
    res = chi_or_fisher(tbl.values); res.update({"test": "S. aureus vs Source", "n": int(tbl.values.sum())})
    results.append(res)
    tbl.plot(kind="bar", stacked=True, ax=ax, color=["lightgray", "seagreen"], edgecolor="k")
    ax.set_title("S. aureus by Top 8 Sources")
    ax.set_xlabel("Source"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(["Negative", "Positive"], title="S. aureus", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
    add_pval_annotation(ax, res["p"])
else:
    ax.text(0.5, 0.5, "⚠️ 'S. aureus' or 'Source' missing; skipped S8b.", ha="center", va="center")
    ax.set_title("S8b (skipped)")

# S8c: MRSA vs Top 8 Sources
ax = axs[1,0]
if "Test" in df.columns and "Source" in df.columns:
    mrsa_sub = df[df["Test"] == "MRSA"]
    top_src = mrsa_sub["Source"].value_counts().head(8).index
    mrsa_sub2 = mrsa_sub[mrsa_sub["Source"].isin(top_src)]
    neg_counts = df[(df["Test"] == "MSSA") & (df["Source"].isin(top_src))]["Source"].value_counts().reindex(top_src, fill_value=0)
    pos_counts = mrsa_sub2["Source"].value_counts().reindex(top_src, fill_value=0)
    tbl = pd.DataFrame({0: neg_counts.values, 1: pos_counts.values}, index=top_src)
    res = chi_or_fisher(tbl.values); res.update({"test": "MRSA vs Source", "n": int(tbl.values.sum())})
    results.append(res)
    tbl.plot(kind="bar", stacked=True, ax=ax, color=["tan", "brown"], edgecolor="k")
    ax.set_title("MRSA by Top 8 Sources")
    ax.set_xlabel("Source"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(["MSSA (0)", "MRSA (1)"], title="Methicillin Status", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
    add_pval_annotation(ax, res["p"])
else:
    ax.text(0.5, 0.5, "⚠️ MRSA data or 'Source' missing; skipped S8c.", ha="center", va="center")
    ax.set_title("S8c (skipped)")

# S8d: BV-Status vs Top 4 Specimens
ax = axs[1,1]
if "BV-Status" in df.columns and "Specimen" in df.columns:
    top_spec = df["Specimen"].value_counts().head(4).index
    sub = df[df["Specimen"].isin(top_spec)]
    tbl = pd.crosstab(sub["Specimen"], sub["BV-Status"]).reindex(columns=["BV-N","BV-P","BV-T"], fill_value=0)
    res = chi_or_fisher(tbl.values); res.update({"test": "BV vs Specimen", "n": int(tbl.values.sum())})
    results.append(res)
    tbl.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="k")
    ax.set_title("BV Status by Top 4 Specimens")
    ax.set_xlabel("Specimen"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="BV-Status", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
    add_pval_annotation(ax, res["p"])
else:
    ax.text(0.5, 0.5, "⚠️ 'BV-Status' or 'Specimen' missing; skipped S8d.", ha="center", va="center")
    ax.set_title("S8d (skipped)")

savefig("fig_SUPP_S8_outcomes_vs_source_specimen.png")
print_qc("✔ Completed Supplementary Fig S8.\n")

# ==============================================================================
# SECTION S9: SUPP FIG S9 – Extra Interactions (1×3 canvas)
#   Pairs: Sa×TRICH, Sa×Atrophy, Sa×CD
# ==============================================================================
print_qc("Generating Supplementary Fig S9 (extra interactions) in a 1×3 grid…")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
sns.despine(left=True)

extra_pairs = [("Sa", "TRICH"), ("Sa", "Atrophy"), ("Sa", "CD")]
for idx, (a, b) in enumerate(extra_pairs):
    ax = axs[idx]
    if a not in factors or b not in df.columns:
        ax.text(0.5, 0.5, f"⚠️ '{a}' or '{b}' missing; skipped S9 {b}.", ha="center", va="center")
        ax.set_title(f"{a}×{b} (skipped)")
        continue

    vec_a = factors[a]
    vec_b = df[b]  # 0/1
    tab = pd.crosstab(vec_a, vec_b)
    total = int(tab.values.sum())
    if total < 20:
        ax.text(0.5, 0.5, f"⚠️ Only {total} total; skipped {a}×{b}.", ha="center", va="center")
        ax.set_title(f"{a}×{b} (skipped)")
        continue

    res = chi_or_fisher(tab.values)
    res.update({"test": f"{a}×{b}", "n": total})
    results.append(res)

    tbl2 = pd.crosstab(vec_b, vec_a)
    tbl2.plot(kind="bar", stacked=True, ax=ax, color=["lightblue", "darkblue"], edgecolor="k")
    ax.set_title(f"{a} × {b}")
    ax.set_xlabel(a); ax.set_ylabel("Count")
    ax.legend(title=b, bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    add_pval_annotation(ax, res["p"])
    
savefig("fig_SUPP_S9_extra_interactions.png")
print_qc("✔ Completed Supplementary Fig S9.\n")

# ==============================================================================
# SECTION S10: SUPP FIG S10 – Placeholder (no extra plots)
# ==============================================================================
print_qc("Supplementary Fig S10: placeholder (no extra plots).\n")

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT ALL PLOT DATA TO EXCEL
# ──────────────────────────────────────────────────────────────────────────────
print_qc("Exporting plot data to Excel…")
# --------------------------------------------------------------------------
# Collect all ML stats (Fig 8) into df_stats  BEFORE we start writing Excel
# --------------------------------------------------------------------------
df_stats = pd.concat([pd.DataFrame(roc_summary), pd.DataFrame(results)], ignore_index=True)
factors["Eth"] = factors["Ethnicity"]      # short alias for export sheets

with pd.ExcelWriter(OUT / "plot_data.xlsx") as writer:
    # Main Fig 1: Age histogram and counts
    if "Age" in df.columns:
        df[["Age"]].dropna().value_counts().rename("count").to_frame().to_excel(writer, sheet_name="Fig1_Age")
    if "BV-Status" in df.columns:
        df["BV-Status"].value_counts().rename("count").to_frame().to_excel(writer, sheet_name="Fig1_BV_Status")
    if "Specimen" in df.columns:
        df["Specimen"].value_counts().rename("count").to_frame().to_excel(writer, sheet_name="Fig1_Specimen")
    if "Source" in df.columns:
        df["Source"].value_counts().head(20).rename("count").to_frame().to_excel(writer, sheet_name="Fig1_Top_Sources")
    if "Ethnicity" in df.columns:
        df["Ethnicity"].value_counts().head(15).rename("count").to_frame().to_excel(writer, sheet_name="Fig1_Ethnicity")

    # Main Fig 2: factors × BV
    # (we’ll reuse the same crosstab we plotted)
    ft = pd.crosstab(df["Test"], df["BV-Status"]).reindex(
        index=["S. aureus","MRSA","MSSA","AST"], columns=["BV-N","BV-P","BV-T"], fill_value=0
    )
    ft.to_excel(writer, sheet_name="Fig2_all_factors")

    # Main Fig 3: AgeBin & Eth, then Pathology
    if "AgeBin" in df.columns:
        pd.crosstab(df["AgeBin"], df["BV-Status"]).to_excel(writer, sheet_name="Fig3_AgeBin")
    if "Ethnicity" in df.columns:
        pd.crosstab(df["Ethnicity"], df["BV-Status"]).to_excel(writer, sheet_name="Fig3_Ethnicity")
    # pathology flags
    for c in cyto_cols:
        if c in df.columns:
            pd.crosstab(df[c], df["BV-Status"]).to_excel(writer, sheet_name=f"Fig3_{c}")

    # Main Fig 4: BV vs infections & S. aureus co‐occurrence
    for ip in infection_cols:
        if ip in df.columns:
            pd.crosstab(df[ip], df["BV-Status"]).to_excel(writer, sheet_name=f"Fig4_BV_{ip}")
    sa = df[df["Test"]=="S. aureus"]
    for ip in infection_cols:
        if ip in sa.columns:
            sa[[ip]].value_counts().rename("count").to_frame().to_excel(
                writer, sheet_name=f"Fig4_Sa_{ip}"
            )

    # Main Fig 5: composite heatmap data (the “heat” DataFrame)
    heat.to_excel(writer, sheet_name="Fig5_heatmap_values")

    # Main Fig 6: interaction tables
    # we merged MRSA×BV & Sa×BV into one sheet; the others separately
    iv = pd.crosstab(factors["MRSA"], factors["BV"])
    iv.index.name = "MRSA"        # name the rows
    iv.columns.name = "BV"        # name the columns

    iv.to_excel(writer, sheet_name="Fig6_MRSAxBV")
    iv2 = pd.crosstab(factors["S. aureus"], factors["BV"])
    iv2.index.name = "Sa"
    iv2.columns.name = "BV"

    iv2.to_excel(writer, sheet_name="Fig6_Sa_xBV")
    pd.crosstab(factors["MRSA"], factors["AgeBin"]).to_excel(writer, sheet_name="Fig6_MRSAxAgeBin")
    pd.crosstab(factors["S. aureus"], factors["Eth"]).to_excel(writer, sheet_name="Fig6_Sa_xEth")

    # Main Fig 7: monthly counts
    monthly = df.groupby(df["Date-Collected"].dt.to_period("M"))
    for label, fn in {
        "BV": lambda d: (d["BV-Status"]=="BV-P").sum(),
        "Sa": lambda d: ((d["Test"]=="S. aureus")&(d["Result"]=="P")).sum(),
        "MRSA": lambda d: (d["Test"]=="MRSA").sum(),
        "MSSA": lambda d: (d["Test"]=="MSSA").sum(),
    }.items():
        pd.DataFrame(monthly.apply(fn), columns=[f"{label}_count"]).to_excel(writer,
            sheet_name=f"Fig7_{label}"
        )

    # Main Fig 8: ROC AUCs & feature importances
    df_stats.to_excel(writer, sheet_name="Fig8_stats")
    for name, fi in importances.items():
        fi.head(10).to_excel(writer, sheet_name=f"Fig8_FI_{name}")

    # After dumping Fig1–Fig8 sheets:
    # Supplementary S1–S9
    # ==============================================================================
# BUILD supplementary_tables FOR S2–S9 PANELS
# ==============================================================================

supplementary_tables = {}

# S2 example: add crosstabs or frequency counts from S2 panels
try:
    # S. aureus (+) by AgeBin, Ethnicity, BV-Status, TRICH
    sa_pos = df[(df["Test"]=="S. aureus") & (df["Result"]=="P")]
    supplementary_tables["S2_AgeBin"] = pd.crosstab(sa_pos["AgeBin"].astype(str), sa_pos["Result"]=="P")
    supplementary_tables["S2_Ethnicity"] = pd.crosstab(sa_pos["Ethnicity"].astype(str), sa_pos["Result"]=="P")
    supplementary_tables["S2_BV_Status"] = pd.crosstab(sa_pos["BV-Status"].astype(str), sa_pos["Result"]=="P")
    supplementary_tables["S2_TRICH"] = pd.crosstab(sa_pos["TRICH"].astype(str), sa_pos["Result"]=="P")
except Exception as e:
    print_qc(f"⚠️ Error adding S2 tables: {e}")

# S3 example: AST vs AgeBin, Ethnicity, BV-Status
try:
    ast_df = df[df["Test"]=="AST"]
    supplementary_tables["S3_AgeBin"] = pd.crosstab(ast_df["AgeBin"].astype(str), ast_df["Result"].notna())
    supplementary_tables["S3_Ethnicity"] = pd.crosstab(ast_df["Ethnicity"].astype(str), ast_df["Result"].notna())
    supplementary_tables["S3_BV_Status"] = pd.crosstab(ast_df["BV-Status"].astype(str), ast_df["Result"].notna())
except Exception as e:
    print_qc(f"⚠️ Error adding S3 tables: {e}")

# S4 example: Cytology volcano DataFrame
try:
    if 'vol_df' in locals():
        supplementary_tables["S4_volcano"] = vol_df
except Exception as e:
    print_qc(f"⚠️ Error adding S4 volcano: {e}")

# S6 example: binned Ct-scores (assuming ct_df exists)
try:
    if 'ct_df' in locals() and 'Ct -score' in ct_df:
        ct_df['Ct_bin'] = pd.cut(ct_df['Ct -score'], bins=[0,20,30,40], labels=['Low','Medium','High'])
        supplementary_tables["S6_Ct_bin_counts"] = ct_df['Ct_bin'].value_counts().to_frame("Count")
except Exception as e:
    print_qc(f"⚠️ Error adding S6 Ct-bin: {e}")

# S7 example: add any additional S7 panel tables here
# supplementary_tables["S7_..."] = your_S7_df

# S9 example: extra interactions
try:
    flag = ((df["Test"]=="S. aureus") & (df["Result"]=="P")).map({False:"Absent", True:"Present"})
    for col in ["TRICH", "Atrophy", "CD"]:
        if col in df.columns:
            tab = pd.crosstab(flag, df[col].astype(str))
            supplementary_tables[f"S9_{col}"] = tab
except Exception as e:
    print_qc(f"⚠️ Error adding S9 tables: {e}")

# Add other S1–S9 DataFrames as you build them...

# ==============================================================================
# EXPORT: Supplementary Figures S1–S9 to Excel (split files if needed)
# ==============================================================================

print_qc("Exporting ALL supplementary plot data to Excel…")

MAX_SHEETS_PER_FILE = 31
sheets = list(supplementary_tables.items())

if not sheets:
    print_qc("⚠️ No supplementary tables found for export.")
else:
    n_files = (len(sheets) + MAX_SHEETS_PER_FILE - 1) // MAX_SHEETS_PER_FILE
    for i in range(n_files):
        start = i * MAX_SHEETS_PER_FILE
        end = min((i+1) * MAX_SHEETS_PER_FILE, len(sheets))
        file_suffix = f"_part{i+1}" if n_files > 1 else ""
        fname = OUT / f"plot_data_supplementary{file_suffix}.xlsx"
        with pd.ExcelWriter(fname) as writer:
            for sheet_name, df_tab in sheets[start:end]:
                sheet_name_trunc = str(sheet_name)[:31]
                df_tab.to_excel(writer, sheet_name=sheet_name_trunc)
        print_qc(f"✔ Supplementary tables written to: {fname}")

print_qc("✔ All supplementary plot data export completed.\n")

# ==============================================================================
# FINAL: FDR correction + save stats_summary.csv
# ==============================================================================
print_qc("Applying FDR correction to all p-values…")
df_stats = pd.DataFrame(results)
if not df_stats.empty:
    pvals = df_stats["p"].values
    padj = multipletests(pvals, method="fdr_bh")[1]
    df_stats["p_adj_FDR"] = padj
    df_stats.to_csv(OUT / "stats_summary.csv", index=False)
    print_qc(f"✔ stats_summary.csv written ({len(df_stats):,} rows).")
else:
    print_qc("⚠️ No tests collected; stats_summary.csv will be empty.\n")

print_qc("✅ All done. Check 'reanalysed_outputs/' for PNGs and stats_summary.csv.")
