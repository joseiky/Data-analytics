#!/usr/bin/env python
# ================================================================
# CF_master_analysis.py
# Comprehensive, parallel workflow for Cystic‑Fibrosis dataset
# (duplicates, longitudinal, associations, ML, PCA, KM, network,
#  Bayesian model, star‑annotated P‑values, within‑sample & patient)
# Last update: 2025‑04‑18
# ================================================================

import os, hashlib, warnings, math, json
import modin.pandas as pd          # parallel DataFrame on Ray
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 600})

from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.nonparametric import kruskalwallis
from joblib import Parallel, delayed

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, roc_auc_score,
                             precision_recall_curve, auc,
                             f1_score, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from lifelines import KaplanMeierFitter, statistics as lf_stats
import networkx as nx
import pymc as pm
import bambi as bmb

# ----------------------------------------------------------------
# 0. File paths & output folders
# ----------------------------------------------------------------
OUTDIR   = "figures"; os.makedirs(OUTDIR, exist_ok=True)
DATA_FILE = "/mnt/data/Dataset.xlsx"
CODE_FILE = "/mnt/data/Encoded_Dataset.xlsx"

# ----------------------------------------------------------------
# 1. Data loading & decoding molecular codes
# ----------------------------------------------------------------
print("➜  Loading data …")
df = pd.read_excel(DATA_FILE)
codes = {s: pd.read_excel(CODE_FILE, sheet_name=s)
         for s in ["Variant_Codebook","mRNA_Codebook",
                   "Protein_Codebook","Common Name_Codebook"]}

def decode(df, sheet, col):
    mapper = dict(zip(codes[sheet]["Code"], codes[sheet]["Label"]))
    return df[col].map(mapper)

df["Variant_decoded"]       = decode(df,"Variant_Codebook","Variant")
df["mRNA_decoded"]          = decode(df,"mRNA_Codebook","mRNA")
df["Protein_decoded"]       = decode(df,"Protein_Codebook","Protein")
df["Common Name_decoded"]   = decode(df,"Common Name_Codebook","Common Name")
df["Abnormal"]              = df["Abnormal Flag"].eq("Y").astype(int)
df["Date Specimen Collected"] = pd.to_datetime(df["Date Specimen Collected"])

# ----------------------------------------------------------------
# 2. Duplicate vs repeat‑test handling
# ----------------------------------------------------------------
print("➜  Resolving duplicates vs repeat tests …")
hash_cols = df.columns.tolist()
df["row_hash"] = (df[hash_cols]
                  .astype(str).agg("|".join, axis=1)
                  .map(lambda x: hashlib.md5(x.encode()).hexdigest()))
is_dupe = df.duplicated("row_hash")
df = df.loc[~is_dupe].copy()
df["repeat_test"] = df.duplicated("MDLNo", keep=False).astype(int)

# ----------------------------------------------------------------
# 3. Patient‑level longitudinal table & timeline plot
# ----------------------------------------------------------------
print("➜  Building longitudinal patient table …")
pat_tbl = (df.sort_values("Date Specimen Collected")
           .groupby("MDL Patient ID")
           .agg(first_date=("Date Specimen Collected","min"),
                last_date =("Date Specimen Collected","max"),
                n_samples =("MDLNo","nunique"),
                n_variants=("Variant_decoded","nunique"))
           .assign(span_days=lambda x: (x.last_date - x.first_date).dt.days))
pat_tbl.to_csv("patient_summary.csv")

# — Patient timelines (random 200) —
print("   Plotting patient timelines …")
fig_pat, axp = plt.subplots(figsize=(8,5))
sample_pts = pat_tbl.sample(n=min(200,len(pat_tbl)), random_state=1)
axp.scatter(sample_pts["first_date"], sample_pts.index, s=10, label="First")
axp.scatter(sample_pts["last_date"],  sample_pts.index, s=10, label="Last", color="orange")
for _,r in sample_pts.iterrows():
    axp.plot([r.first_date,r.last_date],[r.name]*2, lw=0.4, color="gray")
axp.set_title("Patient timelines (random 200)"); axp.legend(); axp.invert_yaxis()
fig_pat.tight_layout(); fig_pat.savefig(f"{OUTDIR}/Supp_patient_timeline.png")

# ----------------------------------------------------------------
# 4. Helper utilities
# ----------------------------------------------------------------
def p_to_stars(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
def chi2_p(a,b):   return sm.stats.Table(pd.crosstab(a,b)).test_nominal_association().pvalue

# ----------------------------------------------------------------
# 5. FIGURE 1 – Demographics + overall outcomes (+ within‑sample)
# ----------------------------------------------------------------
print("➜  Figure 1 …")
fig1, ax = plt.subplots(2,3, figsize=(15,8))
# (a) Age
sns.histplot(df["AGE"].dropna(), bins=30, ax=ax[0,0])
ax[0,0].set_title("Age distribution")
# (b) Sex
sns.countplot(x="Pt Gender", data=df, ax=ax[0,1])
sex_p = chi2_p(df["Pt Gender"], df["Abnormal"])
ax[0,1].set_title(f"Sex ({p_to_stars(sex_p)})")
for c in ax[0,1].containers: ax[0,1].bar_label(c)
# (c) Month volume
vol = df.set_index("Date Specimen Collected").resample("M")["MDLNo"].count()
ax[0,2].plot(vol.index, vol.values, marker="o")
ax[0,2].set_title("Monthly volume")
# (d) Abnormal pie
abn = df.Abnormal.mean(); ax[1,0].pie([abn,1-abn],labels=["Abnormal","Normal"],autopct="%1.1f%%")
ax[1,0].set_title("Overall abnormal")
# (e) Within‑sample variation violin
sns.violinplot(x="repeat_test", y="Abnormal", data=df, ax=ax[1,1])
rep_p = chi2_p(df["repeat_test"], df["Abnormal"])
ax[1,1].set_title(f"Within‑sample abnormal variation ({p_to_stars(rep_p)})")
ax[1,1].set_xticklabels(["Single","Repeat"])
# (f) placeholder free slot
ax[1,2].axis("off")
fig1.tight_layout(); fig1.savefig(f"{OUTDIR}/Fig1_demog_outcomes.png")

# ----------------------------------------------------------------
# 6. FIGURE 2 – Variant landscape, heat‑map, interaction bars, KW age
# ----------------------------------------------------------------
print("➜  Figure 2 …")
top20 = df["Variant_decoded"].value_counts().head(20)
molecular_cols = ["Gene Transcript_decoded","Variant_decoded","mRNA_decoded",
                  "Protein_decoded","Common Name_decoded","Inheritance",
                  "Location","Disease","Zygosity"]
n = len(molecular_cols); p_mat = np.ones((n,n))
for i in range(n):
    for j in range(i+1,n):
        p_mat[i,j]=p_mat[j,i]=chi2_p(df[molecular_cols[i]], df[molecular_cols[j]])

gs = plt.GridSpec(2,3, figure=plt.figure(figsize=(15,8)), hspace=0.45, wspace=0.4)
# (a) Top variants
axV = plt.subplot(gs[0,0]); sns.barplot(y=top20.index, x=top20.values, ax=axV)
axV.set_title("Top‑20 variants"); axV.bar_label(axV.containers[0])
# (b) Interaction heat‑map
axH = plt.subplot(gs[0,1])
sns.heatmap(-np.log10(p_mat), cmap="coolwarm", xticklabels=molecular_cols,
            yticklabels=molecular_cols, ax=axH)
axH.set_title("‑log10 P molecular interactions"); axH.tick_params(labelrotation=45)
# (c) Zygosity
axZ = plt.subplot(gs[0,2]); zyg = df.Zygosity.value_counts()
sns.barplot(x=zyg.index, y=zyg.values, ax=axZ); axZ.bar_label(axZ.containers[0])
axZ.set_title("Zygosity")

# (d) Variant × Sex stacked bars
axVsex = plt.subplot(gs[1,0])
tbl = pd.crosstab(df["Variant_decoded"].where(df["Variant_decoded"].isin(top20.index)),
                  df["Pt Gender"])
tbl.div(tbl.sum(1), axis=0).plot(kind="bar", stacked=True, ax=axVsex, legend=False)
p1 = chi2_p(df["Variant_decoded"].isin(top20.index), df["Pt Gender"])
axVsex.set_title(f"Variant × Sex ({p_to_stars(p1)})")
axVsex.set_ylabel("Proportion"); axVsex.set_xticklabels(axVsex.get_xticklabels(),
                                                        rotation=45, ha="right")

# (e) Variant × Source
axVsrc = plt.subplot(gs[1,1])
top_src = df.Source.value_counts().head(5).index
tbl2 = pd.crosstab(df.loc[df.Source.isin(top_src),"Variant_decoded"]
                        .where(lambda s:s.isin(top20.index)),
                   df.loc[df.Source.isin(top_src),"Source"])
tbl2.div(tbl2.sum(1), axis=0).plot(kind="bar", stacked=True, ax=axVsrc, legend=False)
p2 = chi2_p(df.loc[df.Source.isin(top_src),"Variant_decoded"].isin(top20.index),
            df.loc[df.Source.isin(top_src),"Source"])
axVsrc.set_title(f"Variant × Source ({p_to_stars(p2)})")
axVsrc.set_ylabel("Proportion"); axVsrc.set_xticklabels(axVsrc.get_xticklabels(),
                                                        rotation=45, ha="right")

# (f) Age quartiles KW
axAge = plt.subplot(gs[1,2]); df["age_q"] = pd.qcut(df["AGE"],4,labels=["Q1","Q2","Q3","Q4"])
sns.boxplot(x="age_q",y="AGE",data=df, ax=axAge)
kw_p = kruskalwallis(*[df.loc[df.age_q==q,"AGE"] for q in ["Q1","Q2","Q3","Q4"]]).pvalue
axAge.set_title(f"Age quartiles (KW {p_to_stars(kw_p)})")

plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig2_variants_interactions.png")

# ----------------------------------------------------------------
# 7. FIGURE 3 – PCA + k‑means (sample & patient)
# ----------------------------------------------------------------
print("➜  Figure 3 …")
onehot_var = pd.get_dummies(df["Variant_decoded"].fillna("None"))
scaler = StandardScaler(with_mean=False); Xs = scaler.fit_transform(onehot_var)
pca_s = PCA(n_components=2, random_state=0).fit_transform(Xs)
ks = max(2, min(8, onehot_var.shape[0]//500))
km_s = KMeans(ks, n_init=30, random_state=0).fit(pca_s)

pat_var = df.pivot_table(index="MDL Patient ID",columns="Variant_decoded",
                         values="Abnormal", aggfunc="max", fill_value=0)
Xp = scaler.fit_transform(pat_var)
pca_p = PCA(n_components=2, random_state=0).fit_transform(Xp)
kp = max(2, min(8, pat_var.shape[0]//50))
km_p = KMeans(kp, n_init=30, random_state=0).fit(pca_p)

fig3, ax3 = plt.subplots(2,2, figsize=(11,8))
ax3[0,0].scatter(pca_s[:,0], pca_s[:,1], c=km_s.labels_, s=10)
ax3[0,0].set_title(f"Sample PCA (k={ks})")
ax3[0,1].scatter(pca_p[:,0], pca_p[:,1], c=km_p.labels_, s=15)
ax3[0,1].set_title(f"Patient PCA (k={kp})")
ax3[1,0].bar(range(1,11), PCA().fit(Xs).explained_variance_ratio_[:10])
ax3[1,0].set_title("Sample PCA variance (%)")
ax3[1,1].bar(range(1,11), PCA().fit(Xp).explained_variance_ratio_[:10])
ax3[1,1].set_title("Patient PCA variance (%)")
fig3.tight_layout(); fig3.savefig(f"{OUTDIR}/Fig3_PCA_clusters.png")

# ----------------------------------------------------------------
# 8. FIGURE 4 – Machine‑learning models (LR, RF, XGB + SMOTE)
# ----------------------------------------------------------------
print("➜  Figure 4 …")
features = pd.concat([df[["AGE"]].fillna(df.AGE.median()),
                      pd.get_dummies(df[["Pt Gender","Ethnicity"]]),
                      onehot_var], axis=1).values
y = df.Abnormal.values
from sklearn.model_selection import train_test_split
X_tr,X_te,y_tr,y_te = train_test_split(features,y,stratify=y,test_size=0.25,random_state=42)

def ml_eval(base,name):
    mdl = Pipeline([("sm",SMOTE(random_state=42)),("mdl",base)]).fit(X_tr,y_tr)
    prob = mdl.predict_proba(X_te)[:,1]
    auc_ = roc_auc_score(y_te, prob)
    prec,rec,_ = precision_recall_curve(y_te, prob)
    pr_auc = auc(rec,prec)
    f1  = f1_score(y_te, mdl.predict(X_te))
    return {"name":name,"prob":prob,"auc":auc_,"pr":pr_auc,"f1":f1}

results = [ml_eval(LogisticRegression(max_iter=1000,n_jobs=-1),"LR+SMOTE"),
           ml_eval(RandomForestClassifier(n_estimators=500,n_jobs=-1,class_weight="balanced"),
                   "RF+SMOTE"),
           ml_eval(XGBClassifier(tree_method="hist",n_estimators=500,
                                 learning_rate=0.05,n_jobs=os.cpu_count(),
                                 eval_metric="logloss"),
                   "XGB+SMOTE")]

fig4, ax4 = plt.subplots(2,2, figsize=(11,8))
for r in results:
    RocCurveDisplay.from_predictions(y_te,r["prob"],name=r["name"],ax=ax4[0,0])
ax4[0,0].set_title("ROC curves")

metric_df = pd.DataFrame([{k:v for k,v in r.items() if k in ["name","auc","pr","f1"]}
                          for r in results]).set_index("name")
metric_df.plot(kind="bar", ax=ax4[0,1]); ax4[0,1].set_ylim(0,1)
ax4[0,1].set_title("Model metrics")
for k in range(2): ax4[1,k].axis("off")
fig4.tight_layout(); fig4.savefig(f"{OUTDIR}/Fig4_ML.png")

# ----------------------------------------------------------------
# 9. Sensitivity analyses (missing variant, 'NP' ethnicity)
# ----------------------------------------------------------------
print("➜  Sensitivity analyses …")
for label,df_sens in {"NoMissingVariant": df.dropna(subset=["Variant_decoded"]),
                      "NoNPethnicity":   df[df.Ethnicity!="NP"]}.items():
    p = chi2_p(df_sens["Variant_decoded"].isin(top20.index),
               df_sens["Abnormal"])
    print(f"  χ² Variant vs Abnormal ({label}) P={p:.3g}")

# ----------------------------------------------------------------
# 10. Kaplan‑Meier survival curves
# ----------------------------------------------------------------
print("➜  Kaplan‑Meier survival …")
kmf = KaplanMeierFitter()
time = pat_tbl.span_days.clip(lower=0)+0.1
event = (pat_tbl.n_variants>1).astype(int)
sex = df.drop_duplicates("MDL Patient ID").set_index("MDL Patient ID")["Pt Gender"]

figKM, axKM = plt.subplots(figsize=(6,5))
for grp in sex.unique():
    ix = sex==grp
    kmf.fit(time[ix], event[ix], label=grp).plot(ci_show=False, ax=axKM)
logrank_p = lf_stats.logrank_test(time[sex=="Female"],time[sex=="Male"],
                                  event[sex=="Female"],event[sex=="Male"]).p_value
axKM.set_title(f"Time‑to‑≥2 variants (log‑rank P={p_to_stars(logrank_p)})")
figKM.tight_layout(); figKM.savefig(f"{OUTDIR}/Supp_KM.png")

# ----------------------------------------------------------------
# 11. Variant co‑occurrence network
# ----------------------------------------------------------------
print("➜  Co‑occurrence network …")
co_mat = (onehot_var.T @ onehot_var); np.fill_diagonal(co_mat.values,0)
G = nx.from_pandas_adjacency(co_mat)
H = nx.Graph(); H.add_weighted_edges_from([(u,v,d['weight'])
                                           for u,v,d in G.edges(data=True) if d['weight']>=10])
pos = nx.spring_layout(H, k=0.3, seed=42)
plt.figure(figsize=(8,8))
nx.draw(H,pos,node_size=50,width=[d['weight']/10 for *_,d in H.edges(data=True)],
        with_labels=False)
plt.title("Variant co‑occurrence network (≥10 co‑tests)")
plt.savefig(f"{OUTDIR}/Supp_network.png", dpi=600)

# ----------------------------------------------------------------
# 12. Bayesian hierarchical model (optional, may take minutes)
# ----------------------------------------------------------------
print("➜  Bayesian hierarchical model …")
try:
    df_b = df[["MDL Patient ID","Variant_decoded","Abnormal"]].dropna()
    df_b = df_b[df_b.Variant_decoded.isin(top20.index)]
    m = bmb.Model("Abnormal ~ 1 + (1|`Variant_decoded`) + (1|`MDL Patient ID`)",
                  data=df_b, family="bernoulli")
    idata = m.fit(draws=1000, chains=2, target_accept=0.9)
    m.save("bayes_model.bmr")
except Exception as e:
    print("   Bayesian step skipped:", e)

# ----------------------------------------------------------------
# 13. Methods file
# ----------------------------------------------------------------
print("➜  Writing methods.md …")
with open("methods.md","w") as f:
    f.write("# Methods – Comprehensive CF Variant Analysis\n\n")
    f.write("* Environment: Python 3.10, Modin 0.28 (Ray), statsmodels 0.14, "
            "scikit‑learn 1.5, imbalanced‑learn 0.12, xgboost 2.0, lifelines 0.20, "
            "networkx 3, PyMC 4, Bambi 0.13.*\n\n")
    f.write("Workflow steps: data cleaning, duplicate resolution, repeat‑test flag, "
            "longitudinal patient table (saved as `patient_summary.csv`), "
            "demographic & outcome plots (Fig 1), molecular interaction heat‑map "
            "and association/interaction bars with χ²/KW P‑value annotations (Fig 2), "
            "PCA + k‑means clustering on samples and patients (Fig 3), "
            "machine‑learning prediction models with SMOTE balancing (Fig 4), "
            "sensitivity analyses (variant‑missing, NP ethnicity), Kaplan‑Meier survival "
            "curves (Supp KM), variant co‑occurrence network graph (Supp network), "
            "and an optional Bayesian hierarchical model capturing patient and variant "
            "random effects (saved as `bayes_model.bmr`).\n\n"
            "All graphics exported at 600 dpi under the `figures/` directory.\n")

print("✓ ALL analyses complete – check 'figures/' & 'methods.md'")
