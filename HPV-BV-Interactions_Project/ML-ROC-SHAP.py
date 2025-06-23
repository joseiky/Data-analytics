import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.multitest import multipletests

# =============================================================================
# 1. LOAD & PREPROCESS
# =============================================================================
df = pd.read_excel('Table S1.xlsx', sheet_name=0)

# Map BV to binary: Positive/Transitional=1, Negative=0
# NEW: group BV-P and BV-T as “non-negative” (1), only BV-N as 0
bv_map = {'BV-P':'BV Positive', 'BV-N':'BV Negative', 'BV-T':'Transitional BV'}
df['BV Status Label'] = df['BV Status'].map(bv_map)
df['BV_binary'] = df['BV Status'].map({'BV-P':1, 'BV-T':1, 'BV-N':0}).astype(int)


# Cytology: any ECA or RCC => positive (1), NILM => negative (0)
cyto_cols = ['ECA:ASCUS','ECA:ASC-H','ECA:LSIL','ECA:HSIL','ECA:AGC','RCC']
df['Cyto_binary'] = (df[cyto_cols].fillna(0).sum(axis=1)>0).astype(int)

# Features: Age, hrHPV, lactobacilli, pathogens, HPV types
feature_cols = ['Pt. Age','hrHPV Result'] + \
    ['L. iners ','L. gasseri ','L. jensenii ','L. crispatus '] + \
    ['F. vaginae ','G. vaginalis ','BVAB-2 ',
     'Megasphaera sp.Type 1 ','Megasphaera sp. Type 2 '] + \
    ['HPV 16 ','HPV 18 ','HPV 31 ','HPV 33 ','HPV 35 ',
     'HPV 39 ','HPV 45 ','HPV 51 ','HPV 52 ',
     'HPV 56 ','HPV 58 ','HPV 59 ','HPV 68 ']

X = df[feature_cols].fillna(0)
y_BV = df['BV_binary']
y_CY = df['Cyto_binary']

# Split once, re-use splits for both outcomes
X_train, X_test, yBV_train, yBV_test = train_test_split(
    X, y_BV, test_size=0.25, stratify=y_BV, random_state=42)
_, _, yCY_train, yCY_test = train_test_split(
    X, y_CY, test_size=0.25, stratify=y_CY, random_state=42)

# =============================================================================
# 2. DEFINE MODELS
# =============================================================================
models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
}

# =============================================================================
# 3. FEATURE IMPORTANCE + ROC
# =============================================================================
sns.set_style('whitegrid')

def plot_feature_importance(ax, importances, title):
    idxs = np.argsort(importances)[::-1][:12]
    sns.barplot(
        x=importances[idxs], y=np.array(feature_cols)[idxs],
        color='tab:orange', edgecolor='k', ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.tick_params(labelsize=10)

def plot_roc(ax, y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(labelsize=10)

# Create figures
fig1, axes1 = plt.subplots(2, 4, figsize=(20,10), dpi=150,
                           gridspec_kw={'wspace':0.4,'hspace':0.4})
fig1.suptitle('Figure 5A: Top 12 Feature Importances (BV & Cytology)', fontsize=16, fontweight='bold')

fig2, axes2 = plt.subplots(1, 2, figsize=(12,6), dpi=150, 
                           gridspec_kw={'wspace':0.3})
fig2.suptitle('Figure 5B: ROC Curves (BV & Cytology)', fontsize=16, fontweight='bold')

# Loop over models & outcomes
for col, (name, model) in enumerate(models.items()):
    # BV
    model.fit(X_train, yBV_train)
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    else:
        imp = np.abs(model.coef_)[0]
    plot_feature_importance(axes1[0, col], imp, f'{name} - BV')

    # ROC BV
    proba = model.predict_proba(X_test)[:,1]
    plot_roc(axes2[0], yBV_test, proba, 'BV Status')

    # Cytology
    model.fit(X_train, yCY_train)
    # FI
    if hasattr(model, 'feature_importances_'):
        imp2 = model.feature_importances_
    else:
        imp2 = np.abs(model.coef_)[0]
    plot_feature_importance(axes1[1, col], imp2, f'{name} - Cytology')

    # ROC Cytology
    proba2 = model.predict_proba(X_test)[:,1]
    plot_roc(axes2[1], yCY_test, proba2, 'Cytology')

# Panel letters
for ax, letter in zip(axes1.flatten(), list('ABCDEFGH')[:8]):
    ax.text(-0.1, 1.05, letter, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top')

axes2[0].text(-0.15, 1.05, 'A', transform=axes2[0].transAxes, fontsize=18, fontweight='bold')
axes2[1].text(-0.15, 1.05, 'B', transform=axes2[1].transAxes, fontsize=18, fontweight='bold')

fig1.savefig('Figure5_FeatureImportances.png', bbox_inches='tight')
fig2.savefig('Figure5_ROC.png', bbox_inches='tight')

# =============================================================================
# 4. SHAP ANALYSIS (for tree models)
# =============================================================================
import matplotlib.pyplot as plt

# Use test set for SHAP
X_sample = X_test.copy()

for name in ['XGBoost','Random Forest']:
    model = models[name]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Beeswarm
    shap.summary_plot(
        shap_values, X_sample,
        plot_type='dot',
        show=True  # let it draw its own figure
    )
    plt.title(f'{name} SHAP Beeswarm (BV)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'Figure5_SHAP_{name}_beeswarm.png', dpi=300)
    plt.close()

    # Bar
    shap.summary_plot(
        shap_values, X_sample,
        plot_type='bar',
        show=True
    )
    plt.title(f'{name} SHAP Feature Importance (BV)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'Figure5_SHAP_{name}_bar.png', dpi=300)
    plt.close()