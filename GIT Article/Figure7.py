#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WP-7: BV–Pathogen Interaction Analysis
Generates:
  - Figure 7A: Grouped barplot of top pathogens by BV status
  - Figure 7B: Heatmap of pathogen × BV contingency with p-values
  - Figure 7C: Example stratified barplots by gender
Also writes CSVs:
  - pathogen_prevalence.csv
  - chi2_pvalues.csv
  - contingency_counts.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns

#――――――――――――――――――――――
# 1. Load & subset
#――――――――――――――――――――――
# Read GIT panel and filter out NT
git = pd.read_excel('GIT-No-toxins.xlsx')
git = git[git['BV-Status'] != 'NT']

# Optional: toxin data
cd   = pd.read_excel('CD-Toxins.xlsx')
ecol = pd.read_excel('Ecoli-Shigella-toxins.xlsx')

# Combine all pathogen records
all_data = pd.concat([git, cd, ecol], ignore_index=True, sort=False)
# Focus on presence/absence
all_data['Present'] = all_data['Result'] == 'P'

#― ―――――――――――――――――――――
# 2. Prevalence by BV status
#――――――――――――――――――――――
# Count positives per pathogen & BV status
grouped = (
    all_data
    .groupby(['Test','BV-Status'])['Present']
    .sum()
    .unstack(fill_value=0)
)

# Total samples per BV-Status (for percentages)
totals = all_data.groupby('BV-Status')['MDLNo'].nunique()

# % prevalence
prev_pct = grouped.div(totals, axis=1) * 100

# Select top N pathogens by overall positive count
topN = 15
tot_counts = grouped.sum(axis=1).sort_values(ascending=False)
top_pathogens = tot_counts.head(topN).index
grouped = grouped.loc[top_pathogens]
prev_pct = prev_pct.loc[top_pathogens]

# Save prevalence CSVs
prev_pct.to_csv('pathogen_prevalence_percent.csv')
grouped.to_csv('contingency_counts.csv')

# ――――――――――――――――――――――
# 3. Statistical testing (robust)
# ――――――――――――――――――――――
pvals = {}
for pathogen in top_pathogens:
    sub = all_data[all_data['Test'] == pathogen]
    ct = pd.crosstab(sub['BV-Status'], sub['Present'])
    # ensure both False/True columns exist
    ct = ct.reindex(index=prev_pct.columns, columns=[False, True], fill_value=0)

    # if it's a simple 2×2, use Fisher’s exact
    if ct.shape == (2, 2):
        _, p = fisher_exact(ct.values)
    else:
        try:
            _, p, _, exp = chi2_contingency(ct.values)
        except ValueError:
            # zero expected‐frequency case
            p = np.nan
            print(f"⚠️ Warning: zero expected freq for '{pathogen}', setting p-value=NaN")

    pvals[pathogen] = p

# write out
pd.Series(pvals, name='p_value').to_csv('chi2_pvalues.csv')

#――――――――――――――――――――――
# 4. Figure 7A: Barplot of prevalence %
#――――――――――――――――――――――
sns.set(style='whitegrid')
plt.figure(figsize=(10,6), dpi=300)
prev_pct.plot(kind='bar', edgecolor='k')
plt.ylabel('Prevalence (%)')
plt.title('Figure 7A: Pathogen Prevalence by BV Status')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Figure7A.png', dpi=300)

#――――――――――――――――――――――
# 5. Figure 7B: Heatmap + p-values
#――――――――――――――――――――――
plt.figure(figsize=(8,10), dpi=300)
ax = sns.heatmap(grouped, annot=True, fmt='d', cmap='Blues', cbar_kws={'label':'Count'})
# Annotate p-values to right of each row
for y, pathogen in enumerate(grouped.index):
    ax.text(len(grouped.columns)+0.2, y+0.5, f"p={pvals[pathogen]:.1e}", va='center')
plt.title('Figure 7B: Pathogen × BV Status (counts)\n(p-values to right)')
plt.ylabel('Pathogen'); plt.xlabel('BV Status')
plt.tight_layout()
plt.savefig('Figure7B.png', dpi=300)

#――――――――――――――――――――――
# 6. Figure 7C: Stratified by Gender (example)
#――――――――――――――――――――――
# Compute stratified prevalence %
strat = (
    all_data[all_data['Test'].isin(top_pathogens)]
    .groupby(['Test','Gender','BV-Status'])['Present']
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={'Present':'Pct'})
)

g = sns.catplot(
    data=strat, x='BV-Status', y='Pct',
    col='Test', hue='Gender', kind='bar',
    col_wrap=5, sharey=False, height=3, aspect=1
)
g.set_xticklabels(rotation=45, ha='right')
g.fig.suptitle('Figure 7C: Pathogen Prevalence by BV Status Stratified by Gender', y=1.02)
g.set_axis_labels('', 'Prevalence (%)')
plt.tight_layout()
g.savefig('Figure7C.png', dpi=300)

#――――――――――――――――――――――
# Finished
print("Figures 7A–7C generated; CSV summary files written.")
