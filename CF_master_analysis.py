#!/usr/bin/env python3
# ==============================================================
# CF_master_analysis_v2.py
# Clean, fault‑tolerant, ≤4‑subplot version (pandas only)
# Last update: 2025‑04‑18
# ==============================================================

import os, hashlib, warnings, math, json, sys, logging
from pathlib import Path
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── scientific libs ────────────────────────────────────────────
import statsmodels.api as sm
from scipy.stats import kruskal
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.decomposition   import PCA
from sklearn.cluster         import KMeans
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (roc_auc_score, precision_recall_curve,
                                     auc, f1_score, RocCurveDisplay)
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from imblearn.over_sampling  import SMOTE
from lifelines               import KaplanMeierFitter
from lifelines.statistics    import logrank_test
import networkx as nx

# ── global plotting style ──────────────────────────────────────
sns.set(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 600})

# ── logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="➜ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger()

# ── File paths ────────────────────────────────────────────────
OUTDIR      = Path("figures")
OUTDIR.mkdir(exist_ok=True)
DATA_FILE   = Path("Dataset.xlsx")
CODE_FILE   = Path("Encoded_Dataset.xlsx")

# ==============================================================
# 0. Helper utilities
# ==============================================================
def p_to_stars(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

def chi2_p(a, b) -> float:
    """Nominal association χ² P (crash‑safe)."""
    try:
        return sm.stats.Table(pd.crosstab(a, b)).test_nominal_association().pvalue
    except Exception:
        return np.nan

def safe_read_excel(file: Path, **kw):
    try:
        return pd.read_excel(file, **kw)
    except Exception as e:
        log.error(f"Failed to read {file}: {e}")
        sys.exit(1)

# ==============================================================
# 1. Data loading & code‑book decoding
# ==============================================================
log.info("Loading main dataset …")
df = safe_read_excel(DATA_FILE)

log.info("Loading code‑books …")
sheet_map = {
    "Variant"     : "Variant_Codebook",
    "mRNA"        : "mRNA_Codebook",
    "Protein"     : "Protein_Codebook",
    "Common Name" : "Common Name_Codebook"
}
codebooks = {k: safe_read_excel(CODE_FILE, sheet_name=v) for k, v in sheet_map.items()}

def decode_column(col_name: str, df: pd.DataFrame) -> pd.Series:
    cb = codebooks[col_name]
    mapper = dict(zip(cb["Code"], cb["Label"]))
    return df[col_name].map(mapper)

# Apply decoding (skip if column missing)
for col in sheet_map:
    if col in df.columns:
        df[f"{col}_decoded"] = decode_column(col, df)
    else:
        log.warning(f"Column '{col}' missing – decoding skipped.")

# Standard clean‑ups
df["Abnormal"] = df.get("Abnormal Flag", "").eq("Y").astype(int)
if "Date Specimen Collected" in df.columns:
    df["Date Specimen Collected"] = pd.to_datetime(
        df["Date Specimen Collected"], errors="coerce"
    )

# ==============================================================
# 2. Duplicate vs repeat‑test handling
# ==============================================================
log.info("Resolving duplicates vs repeat tests …")
hash_cols = df.columns.tolist()
df["row_hash"] = (
    df[hash_cols].astype(str).agg("|".join, axis=1)
    .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
)
dupe_mask = df.duplicated("row_hash")
df = df.loc[~dupe_mask].copy()

if "MDLNo" in df.columns:
    df["repeat_test"] = df.duplicated("MDLNo", keep=False).astype(int)
else:
    log.warning("'MDLNo' column missing – repeat_test flag not created.")
    df["repeat_test"] = 0

# ==============================================================
# 3. Patient‑level longitudinal table (summary only)
# ==============================================================
if {"MDL Patient ID", "Date Specimen Collected"}.issubset(df.columns):
    log.info("Building longitudinal patient summary …")
    pat_tbl = (
        df.sort_values("Date Specimen Collected")
          .groupby("MDL Patient ID")
          .agg(first_date=("Date Specimen Collected", "min"),
               last_date=("Date Specimen Collected", "max"),
               n_samples=("MDLNo", "nunique"),
               n_variants=("Variant_decoded", "nunique"))
          .assign(span_days=lambda x: (x.last_date - x.first_date).dt.days)
    )
    pat_tbl.to_csv("patient_summary.csv")
else:
    log.warning("Longitudinal summary skipped – missing patient/date columns.")

# ==============================================================
# 4. FIGURE 1 – Demographics & overall outcomes  (2 × 2)
# ==============================================================
log.info("Figure 1 …")
fig1, ax = plt.subplots(2, 2, figsize=(10, 8))

# (a) Age histogram
if "AGE" in df.columns:
    sns.histplot(df["AGE"].dropna(), bins=30, ax=ax[0, 0], kde=False)
    ax[0, 0].set_title("Age distribution")
else:
    ax[0, 0].set_axis_off()
    log.warning("AGE column missing – subplot (a) blanked.")

# (b) Sex count + χ²
if "Pt Gender" in df.columns:
    sns.countplot(x="Pt Gender", data=df, ax=ax[0, 1])
    sex_p = chi2_p(df["Pt Gender"], df["Abnormal"])
    ax[0, 1].set_title(f"Sex ({p_to_stars(sex_p)})")
    for cont in ax[0, 1].containers:
        ax[0, 1].bar_label(cont, fontsize=8)
else:
    ax[0, 1].set_axis_off()
    log.warning("Pt Gender column missing – subplot (b) blanked.")

# (c) Monthly volume
if "Date Specimen Collected" in df.columns:
    vol = (
        df.set_index("Date Specimen Collected")
          .resample("M")
          .size()
          .rename("Count")
    )
    ax[1, 0].plot(vol.index, vol.values, marker="o")
    ax[1, 0].set_title("Monthly test volume")
    ax[1, 0].tick_params(axis="x", rotation=45)
else:
    ax[1, 0].set_axis_off()

# (d) Abnormal pie
abn_rate = df["Abnormal"].mean()
ax[1, 1].pie([abn_rate, 1 - abn_rate],
             labels=["Abnormal", "Normal"],
             autopct="%1.1f%%", startangle=90)
ax[1, 1].set_title("Overall abnormal")

fig1.tight_layout()
fig1.savefig(OUTDIR / "Fig1_demographics.png")

# ==============================================================
# 5. FIGURE 2 – Variant landscape & interactions  (2 × 2)
# ==============================================================
log.info("Figure 2 …")
top20 = (
    df["Variant_decoded"].value_counts()
    .head(20)
    .rename("Count")
)

molecular_cols = [
    c for c in [
        "Gene Transcript_decoded", "Variant_decoded", "mRNA_decoded",
        "Protein_decoded", "Common Name_decoded", "Inheritance",
        "Location", "Disease", "Zygosity"
    ] if c in df.columns
]

# χ² P‑matrix
n = len(molecular_cols)
p_mat = np.ones((n, n))
for i in range(n):
    for j in range(i + 1, n):
        p_val = chi2_p(df[molecular_cols[i]], df[molecular_cols[j]])
        p_mat[i, j] = p_mat[j, i] = p_val

fig2, ax2 = plt.subplots(2, 2, figsize=(12, 9))

# (a) Top‑20 variants
sns.barplot(y=top20.index, x=top20.values, ax=ax2[0, 0])
ax2[0, 0].set_title("Top‑20 variants")
ax2[0, 0].bar_label(ax2[0, 0].containers[0], fontsize=7)

# (b) Interaction heat‑map (‑log10 P)
sns.heatmap(
    -np.log10(p_mat),
    cmap="coolwarm",
    xticklabels=molecular_cols,
    yticklabels=molecular_cols,
    ax=ax2[0, 1]
)
ax2[0, 1].set_title("‑log₁₀ P molecular interactions")
ax2[0, 1].tick_params(axis="x", rotation=45, ha="right")

# (c) Zygosity counts
if "Zygosity" in df.columns:
    zyg = df["Zygosity"].value_counts()
    sns.barplot(x=zyg.index, y=zyg.values, ax=ax2[1, 0])
    ax2[1, 0].set_title("Zygosity")
    ax2[1, 0].bar_label(ax2[1, 0].containers[0], fontsize=8)
else:
    ax2[1, 0].set_axis_off()

# (d) Age quartiles box‑plot
if "AGE" in df.columns:
    df["age_q"] = pd.qcut(df["AGE"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    sns.boxplot(x="age_q", y="AGE", data=df, ax=ax2[1, 1])
    kw_p = kruskal(
        *[df.loc[df.age_q == q, "AGE"] for q in ["Q1", "Q2", "Q3", "Q4"]]
    ).pvalue
    ax2[1, 1].set_title(f"Age quartiles (KW {p_to_stars(kw_p)})")
else:
    ax2[1, 1].set_axis_off()

fig2.tight_layout()
fig2.savefig(OUTDIR / "Fig2_variant_landscape.png")

# --------------------------------------------------------------
# Supplementary Figure – Variant × Sex / Source (2 panels)
# --------------------------------------------------------------
if {"Variant_decoded", "Pt Gender", "Source"}.issubset(df.columns):
    log.info("Supplementary figure: Variant × Sex / Source …")
    fig_supp, ax_s = plt.subplots(1, 2, figsize=(12, 5))

    # Variant × Sex
    tbl_sex = pd.crosstab(
        df["Variant_decoded"].where(df["Variant_decoded"].isin(top20.index)),
        df["Pt Gender"]
    )
    (tbl_sex.div(tbl_sex.sum(1), axis=0)
             .plot(kind="bar", stacked=True, ax=ax_s[0], legend=False))
    sex_int_p = chi2_p(
        df["Variant_decoded"].isin(top20.index),
        df["Pt Gender"]
    )
    ax_s[0].set_title(f"Variant × Sex ({p_to_stars(sex_int_p)})")
    ax_s[0].set_ylabel("Proportion")
    ax_s[0].set_xticklabels(ax_s[0].get_xticklabels(), rotation=45, ha="right")

    # Variant × Source (top 5 sources)
    top_src = df["Source"].value_counts().head(5).index
    tbl_src = pd.crosstab(
        df.loc[df["Source"].isin(top_src), "Variant_decoded"]
          .where(lambda s: s.isin(top20.index)),
        df.loc[df["Source"].isin(top_src), "Source"]
    )
    (tbl_src.div(tbl_src.sum(1), axis=0)
            .plot(kind="bar", stacked=True, ax=ax_s[1], legend=False))
    src_int_p = chi2_p(
        df.loc[df["Source"].isin(top_src), "Variant_decoded"].isin(top20.index),
        df.loc[df["Source"].isin(top_src), "Source"]
    )
    ax_s[1].set_title(f"Variant × Source ({p_to_stars(src_int_p)})")
    ax_s[1].set_ylabel("Proportion")
    ax_s[1].set_xticklabels(ax_s[1].get_xticklabels(), rotation=45, ha="right")

    fig_supp.tight_layout()
    fig_supp.savefig(OUTDIR / "Supp_variant_interactions.png")

# ==============================================================
# 6. FIGURE 3 – PCA + k‑means  (2 × 2)
# ==============================================================
log.info("Figure 3 …")
if "Variant_decoded" in df.columns:
    onehot = pd.get_dummies(df["Variant_decoded"].fillna("None"), dtype=int)
    scaler = StandardScaler(with_mean=False)
    X_sample = scaler.fit_transform(onehot)

    # ─ Sample‑level PCA
    pca_s = PCA(n_components=2, random_state=0).fit_transform(X_sample)
    k_s = max(2, min(8, onehot.shape[0] // 500))
    km_s = KMeans(k_s, n_init=30, random_state=0).fit(pca_s)

    # ─ Patient‑level PCA
    if "MDL Patient ID" in df.columns:
        pat_var = df.pivot_table(index="MDL Patient ID",
                                 columns="Variant_decoded",
                                 values="Abnormal",
                                 aggfunc="max",
                                 fill_value=0)
        X_pat = scaler.fit_transform(pat_var)
        pca_p = PCA(n_components=2, random_state=0).fit_transform(X_pat)
        k_p = max(2, min(8, pat_var.shape[0] // 50))
        km_p = KMeans(k_p, n_init=30, random_state=0).fit(pca_p)
    else:
        pca_p = km_p = None

    fig3, ax3 = plt.subplots(2, 2, figsize=(11, 9))

    ax3[0, 0].scatter(pca_s[:, 0], pca_s[:, 1], c=km_s.labels_, s=10)
    ax3[0, 0].set_title(f"Sample PCA (k={k_s})")

    if pca_p is not None:
        ax3[0, 1].scatter(pca_p[:, 0], pca_p[:, 1], c=km_p.labels_, s=15)
        ax3[0, 1].set_title(f"Patient PCA (k={k_p})")
    else:
        ax3[0, 1].set_axis_off()

    ax3[1, 0].bar(
        range(1, 11),
        PCA().fit(X_sample).explained_variance_ratio_[:10] * 100
    )
    ax3[1, 0].set_title("Sample PCA variance (%)")

    if pca_p is not None:
        ax3[1, 1].bar(
            range(1, 11),
            PCA().fit(X_pat).explained_variance_ratio_[:10] * 100
        )
        ax3[1, 1].set_title("Patient PCA variance (%)")
    else:
        ax3[1, 1].set_axis_off()

    fig3.tight_layout()
    fig3.savefig(OUTDIR / "Fig3_PCA_clusters.png")
else:
    log.warning("Variant_decoded column missing – PCA skipped.")

# ==============================================================
# 7. FIGURE 4 – Machine‑learning models  (1 × 2)
# ==============================================================
log.info("Figure 4 …")
req_cols = {"AGE", "Pt Gender", "Ethnicity", "Abnormal"}
if req_cols.issubset(df.columns):
    X = pd.concat(
        [
            df[["AGE"]].fillna(df["AGE"].median()),
            pd.get_dummies(df[["Pt Gender", "Ethnicity"]], dtype=int),
            onehot
        ],
        axis=1
    ).values

    y = df["Abnormal"].values
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    def eval_model(base_est, name):
        pipe = Pipeline(
            steps=[("sm", SMOTE(random_state=42)), ("mdl", base_est)]
        ).fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        auc_ = roc_auc_score(y_te, prob)
        prec, rec, _ = precision_recall_curve(y_te, prob)
        pr_auc = auc(rec, prec)
        f1 = f1_score(y_te, pipe.predict(X_te))
        return {"name": name, "prob": prob,
                "auc": auc_, "pr": pr_auc, "f1": f1}

    results = [
        eval_model(LogisticRegression(max_iter=1000, n_jobs=-1), "LR"),
        eval_model(RandomForestClassifier(
            n_estimators=500, n_jobs=-1, class_weight="balanced"), "RF"),
        eval_model(XGBClassifier(
            tree_method="hist", n_estimators=500, learning_rate=0.05,
            n_jobs=os.cpu_count(), eval_metric="logloss"), "XGB")
    ]

    fig4, ax4 = plt.subplots(1, 2, figsize=(12, 6))

    # ROC curves
    for r in results:
        RocCurveDisplay.from_predictions(
            y_te, r["prob"], name=r["name"], ax=ax4[0]
        )
    ax4[0].set_title("ROC curves")

    # Metric bar‑plot
    metric_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k in {"name", "auc", "pr", "f1"}}
         for r in results]
    ).set_index("name")
    metric_df.plot(kind="bar", ax=ax4[1])
    ax4[1].set_ylim(0, 1)
    ax4[1].set_title("Model metrics (AUROC, PRAUC, F1)")
    for container in ax4[1].containers:
        ax4[1].bar_label(container, fmt="%.2f", fontsize=8)

    fig4.tight_layout()
    fig4.savefig(OUTDIR / "Fig4_ML.png")
else:
    log.warning(f"ML step skipped – missing columns: {req_cols - set(df.columns)}")

# ==============================================================
# 8. Kaplan‑Meier (supplementary, 1 panel)
# ==============================================================
if {"MDL Patient ID", "Variant_decoded"}.issubset(df.columns):
    log.info("Kaplan‑Meier …")
    if 'pat_tbl' in globals():
        kmf = KaplanMeierFitter()
        time = pat_tbl["span_days"].clip(lower=0) + 0.1
        event = (pat_tbl["n_variants"] > 1).astype(int)

        # Use sex of first sample per patient
        sex_map = (
            df.drop_duplicates("MDL Patient ID")
              .set_index("MDL Patient ID")["Pt Gender"]
        )
        fig_km, ax_km = plt.subplots(figsize=(6, 5))
        for grp in sex_map.unique():
            idx = sex_map == grp
            kmf.fit(time[idx], event[idx], label=grp).plot(
                ci_show=False, ax=ax_km
            )

        p_logrank = logrank_test(
            time[sex_map == "Female"], time[sex_map == "Male"],
            event[sex_map == "Female"], event[sex_map == "Male"]
        ).p_value
        ax_km.set_title(
            f"Time‑to‑≥2 variants (log‑rank {p_to_stars(p_logrank)})"
        )
        fig_km.tight_layout()
        fig_km.savefig(OUTDIR / "Supp_KM.png")
    else:
        log.warning("Kaplan‑Meier skipped – patient table unavailable.")

# ==============================================================
# 9. Variant co‑occurrence network (supplementary)
# ==============================================================
if "Variant_decoded" in df.columns:
    log.info("Variant co‑occurrence network …")
    co_mat = pd.get_dummies(df["Variant_decoded"]).T @ pd.get_dummies(
        df["Variant_decoded"]
    )
    np.fill_diagonal(co_mat.values, 0)
    G = nx.from_pandas_adjacency(co_mat)
    H = nx.Graph(
        [(u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= 10]
    )
    pos = nx.spring_layout(H, k=0.3, seed=42)
    plt.figure(figsize=(8, 8))
    nx.draw(
        H, pos,
        node_size=50,
        width=[d["weight"] / 10 for *_, d in H.edges(data=True)],
        with_labels=False
    )
    plt.title("Variant co‑occurrence network (≥10 co‑tests)")
    plt.savefig(OUTDIR / "Supp_network.png", dpi=600)
    plt.close()

# ==============================================================
# 10. Methods file
# ==============================================================
log.info("Writing methods.md …")
with open("methods.md", "w") as f:
    f.write("# Methods – Comprehensive Cystic‑Fibrosis Variant Analysis\n\n")
    f.write("*Python 3.10, pandas 2.x, statsmodels 0.14, scikit‑learn 1.5, "
            "imbalanced‑learn 0.12, XGBoost 2.0, lifelines 0.20, networkx 3.*\n\n")
    f.write("Steps: data loading, duplicate removal, longitudinal summary, "
            "demographics & outcomes (Fig 1), variant landscape & interactions "
            "(Fig 2 + supplementary), PCA + k‑means clustering (Fig 3), "
            "machine‑learning models (Fig 4), Kaplan‑Meier survival (Supp KM), "
            "variant co‑occurrence network (Supp network). All graphics exported "
            "at 600 dpi to `figures/`.\n")

log.info("✓ ALL analyses complete – see 'figures/' and 'methods.md'")
