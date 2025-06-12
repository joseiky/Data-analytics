import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_csv("Dataset_1_Cleaned_masterfile.csv", dtype=str)

# Harmonize columns and filter out NT & NILM
df['S_aureus'] = ((df['Test'] == 'S. aureus') & (df['Result'] == 'P')).astype(int)
df['MRSA'] = ((df['Test'] == 'MRSA') & (df['Result'] == 'P')).astype(int)
df['MSSA'] = ((df['Test'] == 'MSSA') & (df['Result'] == 'P')).astype(int)
df['BV_Status'] = df['BV-Status']
df = df[df['BV_Status'].isin(['BV-P', 'BV-T', 'BV-N'])]  # Exclude 'NT'
df['ECA'] = df['ECA'].astype(float) if 'ECA' in df.columns else 0

# Cancerous/precancerous pathologies (update this list if needed)
cancer_pathologies = ['ECA', 'AGC', 'ASC-H', 'LSIL', 'HSIL', 'RCC']
for po in cancer_pathologies:
    if po in df.columns:
        df[po] = df[po].astype(float)
    else:
        df[po] = 0

# Helper for p-value annotation
def get_pval_annot(tab):
    try:
        chi2, p, _, _ = chi2_contingency(tab)
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'
    except:
        return 'ns'

# 1. S. aureus (incl. MRSA/MSSA) & BV-P/BV-T on ECA
df['Sa_combo'] = df[['S_aureus','MRSA','MSSA']].max(axis=1)
combo_ecasub = df[df['BV_Status'].isin(['BV-P','BV-T'])].copy()
tab = pd.crosstab(combo_ecasub['Sa_combo'], combo_ecasub['ECA'])
fig, ax = plt.subplots(figsize=(6,5))
tab.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
pval_annot = get_pval_annot(tab)
ax.set_title('S. aureus (incl. MRSA/MSSA) & BV-P/BV-T → ECA (Cervical Pathology)\np={}'.format(pval_annot))
ax.set_xlabel('S. aureus/MRSA/MSSA Positive (0=Negative, 1=Positive)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('Saureus_MRSA_MSSA_BV_Effect_on_ECA.png', dpi=300)
plt.close()

# 2. Effect of BV-P, BV-T, S. aureus (incl. MRSA/MSSA) on all major cancerous pathologies
fig, axes = plt.subplots(2, 3, figsize=(18,10))
axes = axes.flatten()
for idx, po in enumerate(cancer_pathologies):
    # Exclude NILM, only focus on cancerous pathologies
    if po in df.columns and po != "NILM" and df[po].sum() > 0:
        group_subset = df[((df['BV_Status'].isin(['BV-P', 'BV-T'])) | (df['S_aureus'] == 1))].copy()
        group_subset['Group'] = 'BV-P'
        group_subset.loc[group_subset['BV_Status']=='BV-T', 'Group'] = 'BV-T'
        group_subset.loc[group_subset['S_aureus']==1, 'Group'] = 'S. aureus'
        tab = pd.crosstab(group_subset['Group'], group_subset[po])
        tab.plot(kind='bar', stacked=True, ax=axes[idx], legend=False)
        pval = get_pval_annot(tab)
        axes[idx].set_title(f'BV-P/BV-T & S. aureus → {po} (Pap Outcome)\np={pval}')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Count')
    else:
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('BV_Saureus_MRSA_MSSA_on_CancerousPap_Multipanel.png', dpi=300)
plt.close()

# 3. Effect of ECA and S. aureus on BV (exclude NT)
if 'ECA' in df.columns:
    eca_bv = pd.crosstab([df['ECA'], df['S_aureus']], df['BV_Status'])
    fig, ax = plt.subplots(figsize=(7,5))
    eca_bv.plot(kind='bar', stacked=True, colormap='tab10', ax=ax)
    pval = get_pval_annot(eca_bv)
    ax.set_title('ECA & S. aureus → BV States (BV-P, BV-T, BV-N)\np={}'.format(pval))
    ax.set_xlabel('ECA, S. aureus')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('ECA_Saureus_BV_States.png', dpi=300)
    plt.close()

print("All revised plots saved! Check your working directory for PNG images.")
