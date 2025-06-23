import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ---- LOAD DATA AND MAPPING ----
df = pd.read_excel('Table S1.xlsx', sheet_name=0)

# Map Gender
gender_map = {0: 'Female', 1: 'Male', 2: 'Unknown'}
df['Gender Label'] = df['Pt. Gender'].map(gender_map)

# Map BV status
bv_map = {'BV-P': 'BV Positive', 'BV-N': 'BV Negative', 'BV-T': 'Transitional BV'}
df['BV Status Label'] = df['BV Status'].map(bv_map)

# Map hrHPV if needed
hrhpv_map = {0: 'Negative', 1: 'Positive'}
df['hrHPV Label'] = df['hrHPV Result'].map(hrhpv_map)

# Define columns of interest
lacto_cols = ['L. iners ', 'L. gasseri ', 'L. jensenii ', 'L. crispatus ']
lacto_labels = ['L. iners', 'L. gasseri', 'L. jensenii', 'L. crispatus']

bact_cols = ['F. vaginae ', 'G. vaginalis ', 'BVAB-2 ', 'Megasphaera sp.Type 1 ', 'Megasphaera sp. Type 2 ']
hpv_cols = ['HPV 16 ', 'HPV 18 ', 'HPV 31 ', 'HPV 33 ', 'HPV 35 ', 'HPV 39 ', 'HPV 45 ', 'HPV 51 ', 'HPV 52 ', 'HPV 56 ', 'HPV 58 ', 'HPV 59 ', 'HPV 68 ']
bv_status_col = ['BV Status Label']
cyto_cols = [col for col in df.columns if col.startswith('ECA') or col.startswith('NILM') or col.startswith('RCC')]
outcome_cols = bact_cols + hpv_cols + bv_status_col + cyto_cols
outcome_labels = [c.strip() if c != 'BV Status Label' else 'BV Status' for c in outcome_cols]

# ---- ANALYSIS ----
records = []
for outcome, outlabel in zip(outcome_cols, outcome_labels):
    # Boolean for outcome positive
    if outcome == 'BV Status Label':
        outcome_positive = df[outcome].fillna('') == 'BV Positive'
    elif outcome in cyto_cols:
        # For cytology, positive if 1 (per your Figure-1.py)
        outcome_positive = df[outcome].fillna(0) == 1
    else:
        # Bacteria/HPV: positive if >0
        outcome_positive = df[outcome].fillna(0) > 0

    for spcol, splabel in zip(lacto_cols, lacto_labels):
        lacto_positive = df[spcol].fillna(0) > 0

        # 2x2 for Fisher's and OR
        a = ((lacto_positive) & (outcome_positive)).sum()
        b = ((lacto_positive) & (~outcome_positive)).sum()
        c = ((~lacto_positive) & (outcome_positive)).sum()
        d = ((~lacto_positive) & (~outcome_positive)).sum()

        # Validity check
        if (a+b>0) and (c+d>0) and (a+c>0) and (b+d>0):
            table = np.array([[a, b], [c, d]])
            try:
                orval, pval = fisher_exact(table)
            except:
                orval, pval = np.nan, 1.0
        else:
            orval, pval = np.nan, 1.0

        records.append({
            'Outcome': outlabel,
            'Species': splabel,
            'OR': orval,
            'p': pval,
            'a': a, 'b': b, 'c': c, 'd': d
        })

results = pd.DataFrame(records)
results['q'] = multipletests(results['p'], method='fdr_bh')[1]

# ---- PIVOT FOR PLOTTING ----
heatmap_df = results.pivot(index='Outcome', columns='Species', values='q')
or_df = results.pivot(index='Outcome', columns='Species', values='OR')
heatmap_df = heatmap_df.loc[outcome_labels, lacto_labels]
or_df = or_df.loc[outcome_labels, lacto_labels]

# ---- PLOTTING ----
sns.set_style('white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 13), dpi=400, width_ratios=[1.3,1])

# Panel A: Heatmap of q-values (FDR)
hm = heatmap_df.copy()
mask = hm >= 0.05
sns.heatmap(
    hm, mask=mask, annot=True, fmt=".2g",
    cmap='coolwarm_r', ax=ax1, cbar_kws={'label': 'FDR q-value'},
    linewidths=0.4, linecolor='gray', annot_kws={'fontsize':14}
)
sns.heatmap(
    hm, mask=~mask, annot=False, fmt="",
    cmap='Greys', cbar=False, ax=ax1, linewidths=0.4, linecolor='gray'
)
ax1.set_title("A. FDR-adjusted q-value Heatmap\n(Significant associations colored)", fontsize=22, pad=20)
ax1.set_xlabel("Lactobacillus Species", fontsize=18)
ax1.set_ylabel("Outcome", fontsize=18)
ax1.tick_params(labelsize=14)

# Panel B: Odds Ratios Forest plot
species_order = lacto_labels
outcome_order = list(hm.index)
y_pos = np.arange(len(outcome_order))
colors = ['orange', 'red', 'brown', 'magenta']

for i, sp in enumerate(species_order):
    odds = or_df[sp].values
    qs = heatmap_df[sp].values
    ax2.scatter(
        odds, y_pos + i*0.16 - 0.24,
        s=120, color=colors[i],
        marker='o',
        label=f'{sp} (q<0.05)', edgecolors='k',
        alpha=np.where(qs<0.05, 1.0, 0.2)
    )
    for (x, y, qv) in zip(odds, y_pos + i*0.16 - 0.24, qs):
        if not np.isnan(x):
            ax2.plot([x], [y], 'o', color=colors[i], markersize=10,
                markerfacecolor=colors[i] if qv<0.05 else 'none',
                markeredgecolor=colors[i], alpha=1.0 if qv<0.05 else 0.3, zorder=5)
ax2.axvline(1.0, ls='--', color='gray', lw=2)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(outcome_order, fontsize=14)
ax2.set_xlabel('Odds Ratio', fontsize=18)
ax2.set_xscale('log')
ax2.set_xlim(0.01, 5)
ax2.set_title("B. Odds Ratios: Protective (<1) vs. Risk (>1)\n(Significant associations filled)", fontsize=22, pad=20)
ax2.tick_params(labelsize=14)
ax2.legend(loc='upper right', fontsize=13, title='Lactobacillus Species', title_fontsize=16)

# Panel letters
ax1.text(-0.14, 1.04, 'A', transform=ax1.transAxes, fontsize=30, fontweight='bold', va='top')
ax2.text(-0.18, 1.04, 'B', transform=ax2.transAxes, fontsize=30, fontweight='bold', va='top')

plt.tight_layout(rect=[0,0,1,1])
plt.savefig('Figure4_publication_ready.png', dpi=400)
plt.show()
