# Disease.py

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
import shap
import warnings

# ─── 0. LOAD CODEBOOKS & BUILD MAPPINGS ─────────────────────────────────────────
cb_file    = "Dataset1_Encoded_Dataset.xlsx"
cb_variant = pd.read_excel(cb_file, sheet_name="Variant_Codebook")
cb_mrna    = pd.read_excel(cb_file, sheet_name="mRNA_Codebook")
cb_prot    = pd.read_excel(cb_file, sheet_name="Protein_Codebook")
cb_common  = pd.read_excel(cb_file, sheet_name="Common Name_Codebook")

map_variant = pd.Series(cb_variant.Label.values,    index=cb_variant.Code).to_dict()
map_mrna    = pd.Series(cb_mrna.Label.values,       index=cb_mrna.Code).to_dict()
map_prot    = pd.Series(cb_prot.Label.values,       index=cb_prot.Code).to_dict()
map_common  = pd.Series(cb_common.Label.values,     index=cb_common.Code).to_dict()

# ─── 1. LOAD & PREPARE ─────────────────────────────────────────────────────────────
df = pd.read_excel(
    "Dataset.xlsx",
    header=0,
    usecols="A:V",
    engine="openpyxl"
)
df.columns = df.columns.str.strip()

# ─── 1a. DECODE COLUMNS ────────────────────────────────────────────────────────────
df['Variant_Label']    = df['Variant']   .map(map_variant).fillna(df['Variant'])
df['mRNA_Label']       = df['mRNA']      .map(map_mrna)   .fillna(df['mRNA'])
df['Protein_Label']    = df['Protein']   .map(map_prot)   .fillna(df['Protein'])
df['CommonName_Label'] = df['Common Name'].map(map_common).fillna(df['Common Name'])

# ─── 1b. MAP FLAGS & DISEASE ───────────────────────────────────────────────────────
df['Abnormal'] = df['Abnormal Flag'].map({"A":1, "N":0}).astype(int)
df['is_CF']    = np.where(df['Disease']=="Cystic Fibrosis", 1, 0)

# ─── 1c. AGE GROUPS & DEMOGRAPHICS ────────────────────────────────────────────────
bins   = [0,18,40,60,np.inf]
labels = ['0–17','18–39','40–59','60+']
df['AgeGroup']  = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
df['Sex']       = df['Pt Gender']
df['Ethnicity'] = df['Ethnicity']
df['Source']    = df['Source']

# ─── 2. McNemar heatmap ────────────────────────────────────────────────────────────
ct        = pd.crosstab(df['Abnormal'], df['is_CF']) \
               .reindex(index=[0,1], columns=[0,1], fill_value=0)
p_mcnemar = mcnemar(ct.values).pvalue

# ─── 3. Demographics χ² & Cramer’s V ───────────────────────────────────────────────
def cramers_v(table):
    if table.shape[0]<2 or table.shape[1]<2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    return np.sqrt(chi2/(n*(min(table.shape)-1)))

demo_factors = ['AgeGroup','Sex','Ethnicity','Source']
demo_stats   = {}
for f in demo_factors:
    tbl = pd.crosstab(df[f], df['is_CF'])
    p   = chi2_contingency(tbl)[1] if tbl.size>1 else np.nan
    cv  = cramers_v(tbl)
    demo_stats[f] = (tbl, p, cv)

# ─── 4. SKIP IF ONLY ONE CLASS ────────────────────────────────────────────────────
if df['is_CF'].nunique() < 2:
    print("⚠️ 'is_CF' has only one class; skipping analyses.")
    fig = plt.figure(constrained_layout=True, figsize=(12,6), dpi=600)
    gs  = fig.add_gridspec(2, len(demo_factors), height_ratios=[1,1])

    # heatmap
    ax0 = fig.add_subplot(gs[0, :])
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax0)
    ax0.set_title(f"Lab Abnormal vs Clinical CF\nMcNemar p={p_mcnemar:.3f}")
    ax0.set_ylabel("Abnormal"); ax0.set_xlabel("is_CF")

    # demographics
    for i,f in enumerate(demo_factors):
        ax = fig.add_subplot(gs[1,i])
        tbl_norm = demo_stats[f][0].div(demo_stats[f][0].sum(1),axis=0)
        tbl_norm.plot.bar(stacked=True, ax=ax, legend=False)
        ax.set_title(f"{f} (χ² p={demo_stats[f][1]:.3f})")
        ax.tick_params(axis='x', rotation=90, labelsize=7)

    fig.subplots_adjust(left=0.05, right=0.98)
    plt.savefig("Figure_CF_Disease_Heatmap_Demos.png")
    plt.close()
    sys.exit(0)

# ─── 5. Variant enrichment (top 20) ────────────────────────────────────────────────
top20 = df['Variant_Label'].value_counts().head(20).index.tolist()
variant_results = []
for var in top20:
    flag = (df['Variant_Label']==var)
    tbl  = pd.crosstab(flag, df['is_CF']).reindex(index=[False,True],columns=[0,1],fill_value=0)
    a,b = tbl.loc[True,1], tbl.loc[True,0]
    c,d = tbl.loc[False,1],tbl.loc[False,0]
    OR   = ((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5))
    pval = fisher_exact(tbl.values)[1] if tbl.values.min()<5 else chi2_contingency(tbl.values)[1]
    variant_results.append((var, OR, pval))

var_df = pd.DataFrame(variant_results, columns=['Variant','OR','p'])
var_df['adjP'] = multipletests(var_df['p'], method='fdr_bh')[1]
var_df = var_df.sort_values('adjP').head(20)

# ─── 6. Logistic regression ───────────────────────────────────────────────────────
X_vars = pd.get_dummies(df[demo_factors], drop_first=True)
for var in top20:
    X_vars[f"var_{var}"] = (df['Variant_Label']==var).astype(int)

y    = df['is_CF']
mask = X_vars.notnull().all(axis=1)& y.notnull()
X    = sm.add_constant(X_vars.loc[mask]).astype(float)
y    = y.loc[mask]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        logit = sm.Logit(y, X).fit(disp=False)
    except:
        logit = sm.Logit(y, X).fit_regularized(alpha=1.0, L1_wt=0.0)
coef = logit.summary2().tables[1]
sig  = coef[coef['P>|z|']<0.05]

# ─── 7. ROC comparison ────────────────────────────────────────────────────────────
do_roc = len(np.unique(y))==2
if do_roc:
    auc_abn  = roc_auc_score(y, df.loc[mask,'Abnormal'])
    prob_log = logit.predict(X)
    auc_log  = roc_auc_score(y, prob_log)

# ─── 8. XGBoost + SHAP ───────────────────────────────────────────────────────────
wpos       = ((~y.astype(bool)).sum()/y.sum()) if y.sum()>0 else 1.0
clf        = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=wpos)
clf.fit(X,y)
explainer   = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

# ─── 9. PLOTTING ALL PANELS ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(12,14), dpi=600)
gs  = fig.add_gridspec(
    3, 2,
    width_ratios=[1,1.5],
    hspace=0.6,
    wspace=1.0
)

# shift margins
fig.subplots_adjust(left=0.05, right=0.98)

# A1: heatmap
ax1 = fig.add_subplot(gs[0,0])
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title(f"Lab Abnormal vs Clinical CF\nMcNemar p={p_mcnemar:.3f}")

# A2: variant OR
ax2 = fig.add_subplot(gs[0,1])
ypos = np.arange(len(var_df))
ax2.errorbar(var_df['OR'], ypos, fmt='o')
ax2.set_yticks(ypos)
ax2.set_yticklabels(var_df['Variant'], fontsize=7)
ax2.set_title("Top 20 Variant Enrichment")

# A3a: AgeGroup & Sex (middle-left)
gs_m = gs[1,0].subgridspec(1,2, wspace=0.8)
for idx,f in enumerate(['AgeGroup','Sex']):
    ax = fig.add_subplot(gs_m[0,idx])
    tbl_norm = demo_stats[f][0].div(demo_stats[f][0].sum(1),axis=0)
    tbl_norm.plot.bar(stacked=True, ax=ax, legend=False)
    ax.set_title(f"{f} (χ² p={demo_stats[f][1]:.3f})", fontsize=8)
    ax.tick_params(axis='x', rotation=90, labelsize=6)

# A4: logistic OR (middle-right)
ax4 = fig.add_subplot(gs[1,1])
sig2 = sig.copy()
sig2['OR']   = np.exp(sig2['Coef.'])
sig2['low']  = np.exp(sig2['Coef.'] - 1.96*sig2['Std.Err.'])
sig2['high'] = np.exp(sig2['Coef.'] + 1.96*sig2['Std.Err.'])
ypos2 = np.arange(len(sig2))
labels=[]
for idx in sig2.index:
    if idx.startswith("var_"):
        code = idx.replace("var_","")
        labels.append(map_variant.get(code,code))
    else:
        labels.append(idx)
ax4.errorbar(sig2['OR'], ypos2,
             xerr=[sig2['OR']-sig2['low'], sig2['high']-sig2['OR']], fmt='o')
ax4.set_yticks(ypos2)
ax4.set_yticklabels(labels, fontsize=7)
ax4.tick_params(axis='x', rotation=90)
ax4.set_title("Logistic Model – Significant Predictors")

# A3b: Ethnicity & Source (bottom-left)
gs_b = gs[2,0].subgridspec(1,2, wspace=0.8)
for idx,f in enumerate(['Ethnicity','Source']):
    ax = fig.add_subplot(gs_b[0,idx])
    tbl_norm = demo_stats[f][0].div(demo_stats[f][0].sum(1),axis=0)
    tbl_norm.plot.bar(stacked=True, ax=ax, legend=False)
    ax.set_title(f"{f} (χ² p={demo_stats[f][1]:.3f})", fontsize=8)
    ax.tick_params(axis='x', rotation=90, labelsize=6)

# A5: ROC (bottom-right)
if do_roc:
    ax5 = fig.add_subplot(gs[2,1])
    fpr_ab,tpr_ab,_ = roc_curve(y, df.loc[mask,'Abnormal'])
    fpr_lg,tpr_lg,_ = roc_curve(y, prob_log)
    ax5.plot(fpr_ab, tpr_ab, label=f"Abn AUC={auc_abn:.2f}")
    ax5.plot(fpr_lg, tpr_lg, label=f"Log AUC={auc_log:.2f}")
    ax5.plot([0,1],[0,1],'k--')
    ax5.set_title("ROC Comparison")
    ax5.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("Figure_CF_Disease_Associations.png")
plt.close()

# A6: SHAP summary (separate)
X_shap = X.copy().rename(columns={
    **{f"var_{c}": map_variant.get(c,c) for c in top20}
})
plt.figure(figsize=(8,6), dpi=600)
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.title("SHAP Feature Importance for CF Prediction")
plt.tight_layout()
plt.savefig("Figure_CF_Disease_SHAP.png")
plt.close()

print("✅ All figures saved at 600 dpi.")
