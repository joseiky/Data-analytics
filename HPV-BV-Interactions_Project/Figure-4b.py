import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# --- Load and map as before ---
df = pd.read_excel('Table S1.xlsx', sheet_name=0)
gender_map = {0: 'Female', 1: 'Male', 2: 'Unknown'}
df['Gender Label'] = df['Pt. Gender'].map(gender_map)
bv_map = {'BV-P': 'BV Positive', 'BV-N': 'BV Negative', 'BV-T': 'Transitional BV'}
df['BV Status Label'] = df['BV Status'].map(bv_map)
hrhpv_map = {0: 'Negative', 1: 'Positive'}
df['hrHPV Label'] = df['hrHPV Result'].map(hrhpv_map)

lacto_cols = ['L. iners ', 'L. gasseri ', 'L. jensenii ', 'L. crispatus ']
lacto_labels = ['L. iners', 'L. gasseri', 'L. jensenii', 'L. crispatus']
bact_cols = ['F. vaginae ', 'G. vaginalis ', 'BVAB-2 ', 'Megasphaera sp.Type 1 ', 'Megasphaera sp. Type 2 ']
hpv_cols = ['HPV 16 ', 'HPV 18 ', 'HPV 31 ', 'HPV 33 ', 'HPV 35 ', 'HPV 39 ', 'HPV 45 ', 'HPV 51 ', 'HPV 52 ', 'HPV 56 ', 'HPV 58 ', 'HPV 59 ', 'HPV 68 ']
bv_status_col = ['BV Status Label']
cyto_cols = [col for col in df.columns if col.startswith('ECA') or col.startswith('NILM') or col.startswith('RCC')]
outcome_cols = bact_cols + hpv_cols + bv_status_col + cyto_cols
outcome_labels = [c.strip() if c != 'BV Status Label' else 'BV Status' for c in outcome_cols]

def calc_ci(a, b, c, d, orval):
    if a==0 or b==0 or c==0 or d==0 or np.isnan(orval):
        return np.nan, np.nan
    z = 1.96
    log_or = np.log(orval)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(log_or - z*se)
    ci_upper = np.exp(log_or + z*se)
    return ci_lower, ci_upper

records = []
for outcome, outlabel in zip(outcome_cols, outcome_labels):
    if outcome == 'BV Status Label':
        outcome_positive = df[outcome].fillna('') == 'BV Positive'
    elif outcome in cyto_cols:
        outcome_positive = df[outcome].fillna(0) == 1
    else:
        outcome_positive = df[outcome].fillna(0) > 0

    for spcol, splabel in zip(lacto_cols, lacto_labels):
        lacto_positive = df[spcol].fillna(0) > 0
        a = ((lacto_positive) & (outcome_positive)).sum()
        b = ((lacto_positive) & (~outcome_positive)).sum()
        c = ((~lacto_positive) & (outcome_positive)).sum()
        d = ((~lacto_positive) & (~outcome_positive)).sum()
        if (a+b>0) and (c+d>0) and (a+c>0) and (b+d>0):
            table = np.array([[a, b], [c, d]])
            try:
                orval, pval = fisher_exact(table)
            except:
                orval, pval = np.nan, 1.0
        else:
            orval, pval = np.nan, 1.0
        ci_lo, ci_hi = calc_ci(a, b, c, d, orval)
        records.append({
            'Outcome': outlabel,
            'Species': splabel,
            'OR': orval,
            'CI_lower': ci_lo,
            'CI_upper': ci_hi,
            'p': pval,
            'a': a, 'b': b, 'c': c, 'd': d
        })

results = pd.DataFrame(records)
results['q'] = multipletests(results['p'], method='fdr_bh')[1]

heatmap_df = results.pivot(index='Outcome', columns='Species', values='q').loc[outcome_labels, lacto_labels]
or_df = results.pivot(index='Outcome', columns='Species', values='OR').loc[outcome_labels, lacto_labels]
ci_lo_df = results.pivot(index='Outcome', columns='Species', values='CI_lower').loc[outcome_labels, lacto_labels]
ci_hi_df = results.pivot(index='Outcome', columns='Species', values='CI_upper').loc[outcome_labels, lacto_labels]

sns.set_style('white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 13), dpi=400, width_ratios=[1.3,1])

# Panel A: Heatmap with non-significant cells white/grey
hm = heatmap_df.copy()
sig_mask = hm < 0.05
sns.heatmap(
    hm, mask=~sig_mask, annot=hm.where(sig_mask).round(3), fmt=".3g",
    cmap='coolwarm_r', ax=ax1, cbar_kws={'label': 'FDR q-value'},
    linewidths=0.4, linecolor='gray', annot_kws={'fontsize':14},
    square=False, vmin=0, vmax=0.05
)

# Overlay non-significant as white/grey (use custom color map or 'Greys')
sns.heatmap(
    hm, mask=sig_mask, annot=False, fmt="",
    cmap=sns.color_palette(['white', 'lightgrey'], as_cmap=True), 
    cbar=False, ax=ax1, linewidths=0.4, linecolor='gray'
)
ax1.set_title("A. FDR-adjusted q-value Heatmap\n(Significant associations colored)", fontsize=22, pad=20)
ax1.set_xlabel("Lactobacillus Species", fontsize=18)
ax1.set_ylabel("Outcome", fontsize=18)
ax1.tick_params(labelsize=14)

# Panel B: Forest plot, filled if significant, hollow if not, add CIs
species_order = lacto_labels
outcome_order = list(hm.index)
y_pos = np.arange(len(outcome_order))
colors = ['orange', 'red', 'brown', 'magenta']

for i, sp in enumerate(species_order):
    odds = or_df[sp].values
    qs = heatmap_df[sp].values
    ci_lo = ci_lo_df[sp].values
    ci_hi = ci_hi_df[sp].values
    for j, (orv, qv, lo, hi) in enumerate(zip(odds, qs, ci_lo, ci_hi)):
        if not np.isnan(orv):
            yj = y_pos[j] + i*0.16 - 0.24
            # Plot CI as line
            if not np.isnan(lo) and not np.isnan(hi):
                ax2.plot([lo, hi], [yj, yj], color=colors[i], alpha=0.8, linewidth=3, zorder=2)
            # Plot marker: filled if significant, hollow if not
            ax2.plot(
                orv, yj, marker='o', markersize=12,
                markerfacecolor=colors[i] if qv < 0.05 else 'none',
                markeredgecolor=colors[i], 
                linewidth=2,
                alpha=1.0, zorder=5
            )
ax2.axvline(1.0, ls='--', color='gray', lw=2)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(outcome_order, fontsize=14)
ax2.set_xlabel('Odds Ratio (log scale)', fontsize=18)
ax2.set_xscale('log')
ax2.set_xlim(0.01, 5)
ax2.set_title("B. Odds Ratios: Protective (<1) vs. Risk (>1)\n(Filled=significant, Hollow=non-significant; 95% CI)", fontsize=20, pad=20)
ax2.tick_params(labelsize=14)
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', label=sp, markerfacecolor=col, markeredgecolor=col, markersize=12, lw=0)
    for sp, col in zip(species_order, colors)
]
ax2.legend(handles=legend_elems, loc='upper right', fontsize=13, title='Lactobacillus Species', title_fontsize=16)

# Panel letters
ax1.text(-0.14, 1.04, 'A', transform=ax1.transAxes, fontsize=30, fontweight='bold', va='top')
ax2.text(-0.18, 1.04, 'B', transform=ax2.transAxes, fontsize=30, fontweight='bold', va='top')

plt.tight_layout(rect=[0,0,1,1])
plt.savefig('Figure4_publication_ready_FINAL.png', dpi=400)
plt.show()
