#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure8_full_adjusted.py
WP-8: Supervised ML to Predict BV Status
Outputs:
  - model_metrics.csv
  - Figure8_combined_2x2.png  (panels 8A–8D) with extra left margin for 8C
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. LOAD & PREPARE DATA ───────────────────────────────────────────────────

git = pd.read_excel('GIT-No-toxins.xlsx')
cd  = pd.read_excel('CD-Toxins.xlsx')
eco = pd.read_excel('Ecoli-Shigella-toxins.xlsx')

# filter out 'NT'
git = git[git['BV-Status'] != 'NT']
cd  = cd[cd['BV-Status'] != 'NT']
# eco has no BV-Status

# one-hot encode pathogens
path = pd.crosstab(git['MDLNo'], git['Test']).astype(int)

# C. difficile toxins
cd_pos = cd[cd['Result'] == 'P']
tox_cd = pd.get_dummies(cd_pos['Toxins'], prefix='ToxinCD')
if 'Subtype' in cd_pos.columns:
    tox_cd = tox_cd.join(pd.get_dummies(cd_pos['Subtype'], prefix='SubtypeCD'))
tox_cd.index = cd_pos['MDLNo']
tox_cd = tox_cd.groupby(level=0).max().astype(int)

# E. coli / Shigella toxins
eco_pos = eco[eco['Result'] == 'P']
tox_ec  = pd.get_dummies(eco_pos['Toxins'], prefix='ToxinEC')
if 'Subtype' in eco_pos.columns:
    tox_ec = tox_ec.join(pd.get_dummies(eco_pos['Subtype'], prefix='SubtypeEC'))
tox_ec.index = eco_pos['MDLNo']
tox_ec = tox_ec.groupby(level=0).max().astype(int)

# combine pathogen + toxin features
X_feat = (
    path
    .join(tox_cd, how='outer')
    .join(tox_ec, how='outer')
    .fillna(0)
)

# metadata & target
meta = (
    git[['MDLNo','Age','Gender','Ethnicity','Specimen','Source','BV-Status']]
    .drop_duplicates('MDLNo')
    .set_index('MDLNo')
)
meta['Age'] = pd.to_numeric(meta['Age'], errors='coerce')
y = (meta['BV-Status'] == 'BV Positive').astype(int)

meta_ohe = pd.get_dummies(
    meta[['Gender','Ethnicity','Specimen','Source']],
    drop_first=True
)

# final feature matrix
X = pd.concat([X_feat, meta[['Age']], meta_ohe], axis=1).fillna(0)
X, y = X.align(y, join='inner', axis=0)
X = X.astype(float)

# ─── 2. SPLIT DATA & TRAIN MODELS ────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

models = {
    'Logistic'    : LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0),
    'XGBoost'     : XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
}

for model in models.values():
    model.fit(X_train, y_train)

# ─── 3. EVALUATE & SAVE METRICS ──────────────────────────────────────────────

metrics  = {}
roc_data = {}
cms      = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_data[name] = (fpr, tpr)
    cms[name]      = confusion_matrix(y_test, y_pred)
    metrics[name]  = {
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall'   : recall_score(y_test, y_pred),
        'F1'       : f1_score(y_test, y_pred),
        'AUC'      : roc_auc_score(y_test, y_prob)
    }

pd.DataFrame(metrics).T.to_csv('model_metrics.csv', float_format='%.3f')

# ─── 4. FEATURE IMPORTANCE & SHAP ────────────────────────────────────────────

imp_rf  = pd.Series(models['RandomForest'].feature_importances_, index=X.columns)
imp_xgb = pd.Series(models['XGBoost'].feature_importances_, index=X.columns)
avg_imp = (imp_rf + imp_xgb) / 2
top10   = avg_imp.nlargest(10).index

df_imp = pd.DataFrame({
    'RF' : imp_rf[top10],
    'XGB': imp_xgb[top10]
})

explainer = shap.TreeExplainer(models['XGBoost'])
shap_vals  = explainer.shap_values(X_test)

# ─── 5. PLOT 2×2 GRID WITH EXTRA LEFT MARGIN ─────────────────────────────────

sns.set(style='whitegrid')

# make figure wider to give extra room on left
fig = plt.figure(figsize=(18,12), dpi=300)

# left column 60% wider, more space between columns
outer = fig.add_gridspec(
    2, 2,
    width_ratios=[1.6, 1.0],
    wspace=0.6,
    hspace=0.4
)

# Panel 8A: ROC Curves
axA = fig.add_subplot(outer[0,0])
for name, (fpr, tpr) in roc_data.items():
    axA.plot(fpr, tpr, lw=2, label=f"{name} (AUC={metrics[name]['AUC']:.2f})")
axA.plot([0,1],[0,1],'k--', lw=1)
axA.set(
    xlabel='False Positive Rate',
    ylabel='True Positive Rate',
    title='8A: ROC Curves'
)
axA.legend(loc='lower right', fontsize=8)

# Panel 8B: Confusion Matrices (1×3)
inner = outer[0,1].subgridspec(1,3, wspace=0.3)
for i, (name, cm) in enumerate(cms.items()):
    ax = fig.add_subplot(inner[0,i])
    sns.heatmap(
        cm, annot=True, fmt='d', cbar=False,
        xticklabels=['Neg','Pos'],
        yticklabels=['Neg','Pos'],
        ax=ax
    )
    ax.set(title=name, xlabel='Predicted', ylabel='Actual')

# Panel 8C: Top-10 Feature Importance
axC = fig.add_subplot(outer[1,0])
df_imp.plot(kind='barh', ax=axC)
axC.invert_yaxis()
axC.set(
    title='8C: Top 10 Feature Importance',
    xlabel='Gini / Gain importance'
)

# Panel 8D: Mean(|SHAP|) Bar Summary
axD = fig.add_subplot(outer[1,1])
mean_shap = np.abs(shap_vals).mean(axis=0)
shap_series = pd.Series(mean_shap, index=X.columns).loc[top10]
axD.barh(
    np.arange(len(top10)),
    shap_series.values
)
axD.set_yticks(np.arange(len(top10)))
axD.set_yticklabels(top10)
axD.invert_yaxis()
axD.set(
    title='8D: Mean(|SHAP|) Summary (XGBoost)',
    xlabel='Mean absolute SHAP value'
)

# expand left margin so 8C labels are fully visible
plt.tight_layout()
fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.05)

# save
fig.savefig('Figure8_combined_2x2.png', dpi=300)
