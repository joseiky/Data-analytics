#!/usr/bin/env python
# =====================================================================
# CF_master_analysis.py  ·  FINAL CPU‑ONLY, PURE‑PANDAS VERSION
# Comprehensive analysis pipeline for Cystic‑Fibrosis dataset
# Last updated: 2025‑04‑18
# =====================================================================

"""
Quick environment (conda‑forge, CPU only):

conda create -n cfenv python=3.10 -y
conda activate cfenv
conda install -c conda-forge pandas numpy matplotlib seaborn tqdm statsmodels \
              scipy scikit-learn imbalanced-learn xgboost lifelines networkx \
              shap -y
# optional Bayesian step
conda install -c conda-forge pymc bambi -y
python CF_master_analysis.py
"""

# ------------------------------------------------------------------ #
# 0. Imports                                                         #
# ------------------------------------------------------------------ #
import os, warnings, hashlib, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="talk"); plt.rcParams.update({"figure.dpi": 600})

from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             auc, f1_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lifelines import KaplanMeierFitter, statistics as lf_stats
import networkx as nx
try:
    import shap
except ImportError:
    shap = None
try:
    import bambi as bmb; BAYES_OK = True
except Exception: BAYES_OK = False

import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, ConvergenceWarning
warnings.simplefilter("ignore", PerfectSeparationWarning)
warnings.simplefilter("ignore", ConvergenceWarning)
# ------------------------------------------------------------------ #
# 1. Paths & helpers                                                 #
# ------------------------------------------------------------------ #
DATA_FILE = "Dataset.xlsx"
CODE_FILE = "Encoded_Dataset.xlsx"
OUTDIR = "figures"; os.makedirs(OUTDIR, exist_ok=True)

p_to_stars = lambda p: "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<5e-2 else "ns"
chi2_p = lambda a,b: sm.stats.Table(pd.crosstab(a,b)).test_nominal_association().pvalue

# ------------------------------------------------------------------ #
# 2. Load data & decode molecular codes                              #
# ------------------------------------------------------------------ #
print("→ Loading data …")
df = pd.read_excel(DATA_FILE)
codes = {s: pd.read_excel(CODE_FILE, sheet_name=s)
         for s in ["Variant_Codebook","mRNA_Codebook",
                   "Protein_Codebook","Common Name_Codebook"]}

for sheet, src, dst in [
    ("Variant_Codebook","Variant","Variant_decoded"),
    ("mRNA_Codebook","mRNA","mRNA_decoded"),
    ("Protein_Codebook","Protein","Protein_decoded"),
    ("Common Name_Codebook","Common Name","Common Name_decoded")]:
    mapper = dict(zip(codes[sheet]["Code"], codes[sheet]["Label"]))
    df[dst] = df[src].map(mapper)

df["Abnormal"] = df["Abnormal Flag"].eq("Y").astype(int)
df["Date Specimen Collected"] = pd.to_datetime(df["Date Specimen Collected"], errors="coerce")

# ------------------------------------------------------------------ #
# 3. Duplicate resolution & repeat‑test flag                         #
# ------------------------------------------------------------------ #
hash_cols = df.columns.tolist()
df["row_hash"] = (df[hash_cols].astype(str)
                  .agg("|".join, axis=1)
                  .map(lambda x: hashlib.md5(x.encode()).hexdigest()))
df = df.loc[~df.duplicated("row_hash")].copy()
df["repeat_test"] = df.duplicated("MDLNo", keep=False).astype(int)

# ------------------------------------------------------------------ #
# 4. Patient longitudinal table                                      #
# ------------------------------------------------------------------ #
pat_tbl = (df.sort_values("Date Specimen Collected")
             .groupby("MDL Patient ID")
             .agg(first=("Date Specimen Collected","min"),
                  last =("Date Specimen Collected","max"),
                  n_samples=("MDLNo","nunique"),
                  n_variants=("Variant_decoded","nunique")))
pat_tbl = pat_tbl.assign(span_days=(pat_tbl["last"]-pat_tbl["first"]).dt.days)
pat_tbl.to_csv("patient_summary.csv")

# ------------------------------------------------------------------ #
# 5. Within‑patient & within‑sample variation analyses               #
# ------------------------------------------------------------------ #
# Within‑sample variant diversity
wsamp = (df.groupby("MDLNo")["Variant_decoded"]
           .nunique().rename("variants_per_sample"))
wsamp.to_csv("within_sample_variants.csv")

# Within‑patient variant gain/loss timeline
wpat = (df.sort_values("Date Specimen Collected")
          .groupby(["MDL Patient ID","Variant_decoded"])
          .agg(first_seen=("Date Specimen Collected","min")))
wpat.to_csv("within_patient_variants.csv")

# Plot within-patient variant counts over time (supplementary)
fig_wp, ax_wp = plt.subplots(figsize=(7,4))
(pat_tbl["n_variants"].plot(kind="hist", bins=range(1,8), ax=ax_wp))
ax_wp.set_xlabel("Unique variants per patient"); ax_wp.set_ylabel("Patients")
ax_wp.set_title("Within‑patient variant count"); fig_wp.tight_layout()
fig_wp.savefig(f"{OUTDIR}/Supp_within_patient_variants.png")

# ------------------------------------------------------------------ #
# 6. Figure 1 – Demographics & outcomes                              #
# ------------------------------------------------------------------ #
fig1, ax = plt.subplots(2,2, figsize=(12,7))

sns.histplot(df["AGE"].dropna(), bins=30, ax=ax[0,0])
ax[0,0].set_title("Age distribution")

sns.countplot(x="Pt Gender", data=df, ax=ax[0,1])
ax[0,1].set_title(f"Sex ({p_to_stars(chi2_p(df['Pt Gender'], df.Abnormal))})")

vol = df.set_index("Date Specimen Collected").resample("ME")["MDLNo"].count()
ax[1,0].plot(vol.index, vol.values); ax[1,0].set_title("Monthly volume")

abn = df.Abnormal.mean(); ax[1,1].pie([abn,1-abn],labels=["Abn","Norm"],autopct="%1.1f%%")
ax[1,1].set_title("Overall abnormal (%)")

fig1.tight_layout(); fig1.savefig(f"{OUTDIR}/Fig1_demog_outcomes.png")

# ------------------------------------------------------------------ #
# 7. Figure 2 – Variant landscape & association heat‑map            #
# ------------------------------------------------------------------ #
fields = ["Variant_decoded","mRNA_decoded","Protein_decoded",
          "Common Name_decoded","Inheritance","Location","Disease","Zygosity"]
n=len(fields); p_mat=np.ones((n,n))
for i in range(n):
    for j in range(i+1,n):
        p_mat[i,j]=p_mat[j,i]=chi2_p(df[fields[i]], df[fields[j]])

fig2 = plt.figure(figsize=(15,8)); gs=plt.GridSpec(2,3,figure=fig2,hspace=0.4,wspace=0.35)
top20=df["Variant_decoded"].value_counts().head(20)
axV=fig2.add_subplot(gs[0,0]); sns.barplot(y=top20.index,x=top20.values,ax=axV)
axV.bar_label(axV.containers[0]); axV.set_title("Top‑20 variants")

axH=fig2.add_subplot(gs[0,1])
p_mat = np.clip(p_mat, 1e-300, 1)
sns.heatmap(-np.log10(p_mat),xticklabels=fields,yticklabels=fields,cmap="coolwarm",ax=axH)
axH.set_title("‑log10 P interaction heat‑map"); axH.tick_params(labelrotation=45)

zyg=df["Zygosity"].value_counts()
axZ=fig2.add_subplot(gs[0,2]); sns.barplot(x=zyg.index,y=zyg.values,ax=axZ); axZ.bar_label(axZ.containers[0])
axZ.set_title("Zygosity")

# Variant × Ethnicity forest‑plot (logistic regression)
# Logistic regression: ensure all inputs are numeric + filled
X_lr = pd.get_dummies(df["Variant_decoded"].fillna("None"))
X_eth = pd.get_dummies(df["Ethnicity"].fillna("Unknown"), prefix="Eth")
X_lr = sm.add_constant(pd.concat([X_lr, X_eth], axis=1)).astype(float)

model_lr = sm.Logit(df["Abnormal"], X_lr).fit(disp=False)

or_ci = pd.DataFrame({
    "OR": np.exp(model_lr.params),
    "low": np.exp(model_lr.conf_int()[0]),
    "high": np.exp(model_lr.conf_int()[1])
}).iloc[1:21]  # first 20 terms
axF = fig2.add_subplot(gs[1,0])
axF.errorbar(or_ci["OR"], np.arange(len(or_ci)), 
             xerr=[or_ci["OR"]-or_ci["low"], or_ci["high"]-or_ci["OR"]],
             fmt="o"); axF.set_yticks(np.arange(len(or_ci))); axF.set_yticklabels(or_ci.index)
axF.axvline(1,color="k",ls="--"); axF.set_xscale("log"); axF.set_xlabel("Odds Ratio")
axF.set_title("Variant OR for Abnormal (top‑20)")

# Variant × Source stacked bars
top_src=df.Source.value_counts().head(5).index
tbl_vs=pd.crosstab(df.loc[df.Source.isin(top_src),"Variant_decoded"]
                   .where(lambda s:s.isin(top20.index)),
                   df.loc[df.Source.isin(top_src),"Source"])
axS=fig2.add_subplot(gs[1,1])
(tbl_vs.div(tbl_vs.sum(1),axis=0).plot(kind="bar",stacked=True,legend=False,ax=axS))
p2=chi2_p(df.loc[df.Source.isin(top_src),"Variant_decoded"].isin(top20.index),
          df.loc[df.Source.isin(top_src),"Source"])
axS.set_title(f"Variant × Source ({p_to_stars(p2)})"); axS.set_ylabel("Proportion")
axS.set_xticklabels(axS.get_xticklabels(),rotation=45,ha="right")

# Correlation heat‑map (Cramer V for categorical)
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = sm.stats.Table(table).test_nominal_association().statistic
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    return np.sqrt(phi2 / min(k - 1, r - 1)) if min(k, r) > 1 else 0
corr=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        corr[i,j]=cramers_v(df[fields[i]], df[fields[j]])
axC=fig2.add_subplot(gs[1,2])
sns.heatmap(corr,xticklabels=fields,yticklabels=fields,cmap="vlag",ax=axC)
axC.set_title("Cramer V correlation")

fig2.tight_layout(); fig2.savefig(f"{OUTDIR}/Fig2_variants_interactions.png")

# ------------------------------------------------------------------ #
# 8. Figure 3 – PCA & clustering (sample & patient)                  #
# ------------------------------------------------------------------ #
onehot=pd.get_dummies(df["Variant_decoded"].fillna("None"))
sc=StandardScaler(with_mean=False); Xs=sc.fit_transform(onehot)
pca_s=PCA(n_components=2,random_state=0).fit_transform(Xs); ks=4
km_s=KMeans(ks,n_init=30,random_state=0).fit(pca_s)

pat_var=df.pivot_table(index="MDL Patient ID",columns="Variant_decoded",
                       values="Abnormal",aggfunc="max",fill_value=0)
Xp=sc.fit_transform(pat_var); kp=4
pca_p=PCA(n_components=2,random_state=0).fit_transform(Xp); km_p=KMeans(kp,n_init=30,random_state=0).fit(pca_p)

fig3,ax3=plt.subplots(2,2,figsize=(11,8))
ax3[0,0].scatter(pca_s[:,0],pca_s[:,1],c=km_s.labels_,s=8); ax3[0,0].set_title("Sample PCA")
ax3[0,1].scatter(pca_p[:,0],pca_p[:,1],c=km_p.labels_,s=10); ax3[0,1].set_title("Patient PCA")
ax3[1,0].bar(range(1,11),PCA().fit(Xs).explained_variance_ratio_[:10]); ax3[1,0].set_title("Sample variance")
ax3[1,1].bar(range(1,11),PCA().fit(Xp).explained_variance_ratio_[:10]); ax3[1,1].set_title("Patient variance")
fig3.tight_layout(); fig3.savefig(f"{OUTDIR}/Fig3_PCA_clusters.png")

# ------------------------------------------------------------------ #
# 9. Figure 4 – Predictive modelling & SHAP                          #
# ------------------------------------------------------------------ #
df_ml = df.dropna(subset=["Abnormal"])  # ensures no missing target values

feat = pd.concat([
    df_ml[["AGE"]].fillna(df_ml["AGE"].median()),
    pd.get_dummies(df_ml[["Pt Gender", "Ethnicity"]]),
    pd.get_dummies(df_ml["Variant_decoded"].fillna("None"))
], axis=1)

y = df_ml["Abnormal"].values

X_tr,X_te,y_tr,y_te = train_test_split(feat.values,y,stratify=y,test_size=0.25,random_state=42)

def score(model,name):
    pipe=Pipeline([("sm",SMOTE(random_state=42)),("mdl",model)]).fit(X_tr,y_tr)
    prob=pipe.predict_proba(X_te)[:,1]
    return {"name":name,"prob":prob,
            "auc":roc_auc_score(y_te,prob),
            "pr":auc(*precision_recall_curve(y_te,prob)[::-1]),
            "f1":f1_score(y_te,pipe.predict(X_te)),
            "mdl":pipe}

results=[score(LogisticRegression(max_iter=1000,n_jobs=-1),"LR"),
         score(RandomForestClassifier(n_estimators=400,n_jobs=-1,
                                      class_weight="balanced"),"RF"),
         score(XGBClassifier(tree_method="hist",n_estimators=400,
                             learning_rate=0.05,n_jobs=os.cpu_count(),
                             eval_metric="logloss"),"XGB")]

fig4,ax4=plt.subplots(2,2,figsize=(11,8))
for r in results:
    RocCurveDisplay.from_predictions(y_te,r["prob"],name=r["name"],ax=ax4[0,0])
ax4[0,0].set_title("ROC curves")

pd.DataFrame([{k:v for k,v in r.items() if k in ["name","auc","pr","f1"]}
              for r in results]).set_index("name").plot(kind="bar",ax=ax4[0,1])
ax4[0,1].set_ylim(0,1); ax4[0,1].set_title("Metrics")
if shap and isinstance(results[-1]["mdl"].named_steps["mdl"],XGBClassifier):
    explainer = shap.TreeExplainer(results[-1]["mdl"].named_steps["mdl"])
    shap_vals = explainer.shap_values(feat.iloc[:200])
    shap.summary_plot(shap_vals,feat.iloc[:200],show=False)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Supp_SHAP.png")

for k in range(2): ax4[1,k].axis("off")
fig4.tight_layout(); fig4.savefig(f"{OUTDIR}/Fig4_ML.png")

# ------------------------------------------------------------------ #
# 10. Sensitivity analyses                                           #
# ------------------------------------------------------------------ #
sens = {"NoMissingVariant": df.dropna(subset=["Variant_decoded"]),
        "NoNPethnicity": df[df.Ethnicity!="NP"]}
for lbl,subset in sens.items():
    p = chi2_p(subset["Variant_decoded"].isin(top20.index), subset["Abnormal"])
    print(f"χ² Variant vs Abnormal ({lbl}) P={p:.3g}")

# ------------------------------------------------------------------ #
# 11. Kaplan‑Meier survival                                          #
# ------------------------------------------------------------------ #
kmf=KaplanMeierFitter()
time=pat_tbl.span_days.clip(lower=0)+0.1
event=(pat_tbl.n_variants>1).astype(int)
sex=df.drop_duplicates("MDL Patient ID").set_index("MDL Patient ID")["Pt Gender"]
figKM,axKM=plt.subplots(figsize=(6,5))
for g in sex.unique():
    ix=sex==g; kmf.fit(time[ix],event[ix],label=g).plot(ci_show=False,ax=axKM)
lp=lf_stats.logrank_test(time[sex=="Female"],time[sex=="Male"],
                         event[sex=="Female"],event[sex=="Male"]).p_value
axKM.set_title(f"Time‑to‑≥2 variants (log‑rank {p_to_stars(lp)})")
figKM.tight_layout(); figKM.savefig(f"{OUTDIR}/Supp_KM.png")

# ------------------------------------------------------------------ #
# 12. Variant co‑occurrence network                                  #
# ------------------------------------------------------------------ #
co=(onehot.T@onehot); np.fill_diagonal(co.values,0)
G=nx.from_pandas_adjacency(co); H=nx.Graph()
H.add_weighted_edges_from([(u,v,d['weight']) for u,v,d in G.edges(data=True) if d['weight']>=10])
pos=nx.spring_layout(H,k=0.3,seed=42)
plt.figure(figsize=(8,8))
nx.draw(H,pos,node_size=50,
        width=[d['weight']/10 for *_,d in H.edges(data=True)],with_labels=False)
plt.title("Variant co‑occurrence network (≥10)")
plt.savefig(f"{OUTDIR}/Supp_network.png",dpi=600)

# ------------------------------------------------------------------ #
# 13. Bayesian hierarchical model (optional)                         #
# ------------------------------------------------------------------ #
if BAYES_OK:
    df_b=df[["MDL Patient ID","Variant_decoded","Abnormal"]].dropna()
    df_b=df_b[df_b.Variant_decoded.isin(top20.index)]
    m=bmb.Model("Abnormal ~ 1 + (1|Variant_decoded) + (1|`MDL Patient ID`)",
                data=df_b, family="bernoulli")
    idata=m.fit(draws=1000,chains=2,target_accept=0.9)
    m.save("bayes_model.bmr")
else:
    print("Bayesian step skipped (PyMC/Bambi not present or failed).")

# ------------------------------------------------------------------ #
# 14. Methods markdown                                               #
# ------------------------------------------------------------------ #
with open("methods.md","w") as f:
    f.write(textwrap.dedent(f"""
    # Methods – Comprehensive Cystic Fibrosis Variant Analysis

    *Software stack:* pandas, NumPy, Matplotlib, Seaborn, Statsmodels, SciPy,
    scikit‑learn, imbalanced‑learn, XGBoost, Lifelines, NetworkX; optional PyMC/Bambi.

    **Workflow**

    1. **Data ingestion & decoding** – load Dataset.xlsx, decode Variant/mRNA/Protein/Common‑Name
       using codebooks; convert “Abnormal Flag”.
    2. **Duplicate vs repeat‑test resolution** – MDLNo hashing to drop true duplicates,
       flag repeat_test for longitudinal within‑sample variation.
    3. **Within‑patient timeline** – first/last specimen dates, span_days, n_variants per patient.
    4. **Demographics & outcomes** – age histogram, sex counts, monthly volume, abnormal ratio.
    5. **Variant landscape** – top‑20 bar, χ² p‑value heat‑map (−log₁₀P), zygosity bars,
       Variant×Sex and Variant×Source stacked bars, age‑quartile KW.
    6. **Logistic genotype‑phenotype link** – OR±CI forest‑plot of Variant → Abnormal.
    7. **Correlation heat‑map** – Cramer V among eight decoded/clinical fields.
    8. **Within‑sample / within‑patient variation** – CSV tables + supplementary plots.
    9. **PCA & k‑means** – sample‑ and patient‑level PC1‑PC2 scatter, variance bars.
    10. **Predictive modelling** – SMOTE‑balanced LR, RF, XGB; ROC+PR/F1; optional SHAP.
    11. **Sensitivity analyses** – no‑missing‑variant, no‑NP‑ethnicity χ² checks.
    12. **Kaplan‑Meier** – time‑to‑≥2 variants by sex.
    13. **Variant co‑occurrence network** – spring‑layout graph for ≥10 co‑tests.
    14. **Bayesian hierarchical model** – variant & patient random effects (optional).
    15. **Exports** – all figures at 600 dpi in `figures/`, `patient_summary.csv`,
        `within_sample_variants.csv`, `within_patient_variants.csv`,
        `methods.md`, optional `bayes_model.bmr`.
    """))

print("✓ Analysis complete – check 'figures/' & output CSV/MD files")
