import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch

# --- LOAD DATA ---
df = pd.read_excel('Table S1.xlsx', sheet_name=0)

# Map BV codes for clarity (optional, depending on your data)
df['BV Status'] = df['BV Status'].map({'BV-P':'BV Positive',
                                       'BV-N':'BV Negative',
                                       'BV-T':'Transitional BV'})

# --- SPECIES, MARKERS, AND PANEL LABELS ---
species_cols = ['L. iners ', 'L. gasseri ', 'L. jensenii ', 'L. crispatus ']
species_labels = ['L. iners', 'L. gasseri', 'L. jensenii', 'L. crispatus']
panel_letters = ['A', 'B', 'C', 'D']  # For subplot labeling

bact_cols = [
    'F. vaginae ', 'G. vaginalis ', 'BVAB-2 ',
    'Megasphaera sp.Type 1 ', 'Megasphaera sp. Type 2 '
]
hpv_cols = [
    'HPV 16 ', 'HPV 18 ', 'HPV 31 ', 'HPV 33 ', 'HPV 35 ',
    'HPV 39 ', 'HPV 45 ', 'HPV 51 ', 'HPV 52 ',
    'HPV 56 ', 'HPV 58 ', 'HPV 59 ', 'HPV 68 '
]
comarkers = bact_cols + hpv_cols
outcomes = [col.strip() for col in comarkers]

# --- STATISTICS AND DATA PREP ---
panel_data = {}
species_total_counts = []  # Store denominators for each Lactobacillus species
for sp in species_cols:
    mask = df[sp].fillna(0) > 0
    species_total_counts.append(mask.sum())
    counts = (df[mask][comarkers].fillna(0) > 0).sum()
    pct = counts / mask.sum() * 100 if mask.sum() > 0 else counts * 0
    pvals = []
    for col in comarkers:
        a = ((df[sp].fillna(0)>0) & (df[col].fillna(0)>0)).sum()
        b = ((df[sp].fillna(0)>0) & (df[col].fillna(0)==0)).sum()
        c = ((df[sp].fillna(0)==0) & (df[col].fillna(0)>0)).sum()
        d = ((df[sp].fillna(0)==0) & (df[col].fillna(0)==0)).sum()
        p = fisher_exact([[a,b],[c,d]])[1]
        pvals.append(p)
    reject, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    panel_data[sp] = (counts.astype(int).values, pct.values, pvals, qvals)

# --- PLOTTING ---
sns.set_style('whitegrid')
sns.set_context('talk')

fig, axes = plt.subplots(
    2, 2,
    figsize=(58, 42), dpi=400,
    gridspec_kw={
        'hspace': 0.54,
        'wspace': 0.12,
        'left': 0.06,
        'right': 0.97,
        'top': 0.95,
        'bottom': 0.18
    }
)
axes = axes.flatten()

# Unique color per marker
bar_colors = sns.color_palette("tab20", n_colors=len(outcomes))

# Prepare FDR legend handles
sig_legend_labels = ['*  q<0.05', '**  q<0.01', '***  q<0.001', 'n.s.  q≥0.05']
fdr_handles = [Patch(facecolor='white', edgecolor='none', label=lab)
               for lab in sig_legend_labels]

for idx, (ax, sp, label) in enumerate(zip(axes, species_cols, species_labels)):
    counts, pct, pvals, qvals = panel_data[sp]
    x = np.arange(len(outcomes))

    # Draw bars
    bars = ax.bar(
        x, pct,
        color=bar_colors,
        edgecolor='k',
        linewidth=1.6,
        width=0.99
    )

    # Panel title: include panel letter and denominator for each species
    ax.set_title(
        f"{panel_letters[idx]} {label} (n={species_total_counts[idx]})",
        fontsize=52, fontweight='bold', pad=45
    )
    ax.set_ylabel(
        f'% Present among\n{label}+ samples',
        fontsize=38
    )
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=90, ha='center', fontsize=38)
    ax.tick_params(axis='y', labelsize=24)
    ax.margins(x=0)
    ax.set_xlim(-0.5, len(outcomes)-0.5)

    # Annotate each bar (frequency, percent, significance)
    maxpct = pct.max() if len(pct) > 0 else 0
    for xi, bar in enumerate(bars):
        h = bar.get_height()
        offset = maxpct * 0.01
        y0 = h + offset
        y1 = h + offset * 4
        y2 = h + offset * 7
        if xi % 2 == 1:
            y0 += maxpct * 0.08
            y1 += maxpct * 0.08
            y2 += maxpct * 0.08

        p = pvals[xi]
        q = qvals[xi]
        sig = (
            'n.s.' if p >= 0.05 else
            '***' if q < 0.001 else
            '**' if q < 0.01 else
            '*' if q < 0.05 else 'n.s.'
        )

        ax.text(
            bar.get_x() + bar.get_width()/2, y0,
            f"n={counts[xi]}", ha='center', va='bottom',
            fontsize=28, fontweight='bold'
        )
        ax.text(
            bar.get_x() + bar.get_width()/2, y1,
            f"{pct[xi]:.1f}%", ha='center', va='bottom',
            fontsize=24
        )
        ax.text(
            bar.get_x() + bar.get_width()/2, y2,
            sig, ha='center', va='bottom',
            fontsize=28, color='firebrick'
        )

    # Marker legend only in Panel A
    if idx == 0:
        marker_handles = [
            Patch(facecolor=bar_colors[i], edgecolor='k', label=outcomes[i])
            for i in range(len(outcomes))
        ]
        ax.legend(
            handles=marker_handles,
            title='Marker',
            loc='upper left',
            bbox_to_anchor=(1.14, 1.0),
            fontsize=24,
            title_fontsize=26,
            frameon=True,
            fancybox=True
        )

    # FDR significance legend in every panel
    ax.legend(
        handles=fdr_handles,
        title='Significance\n(FDR q)',
        loc='upper right',
        fontsize=24,
        title_fontsize=26,
        frameon=True,
        fancybox=True,
        borderaxespad=0.8
    )

    sns.despine(ax=ax)
    ax.grid(axis='y', linestyle='--', linewidth=1.2)

# Tight layout and bottom margin for x-axis labels
plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
fig.subplots_adjust(bottom=0.17)

# Save and show
plt.savefig('Figure3_publication_ready.png', dpi=400)