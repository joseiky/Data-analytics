import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np

sns.set_style('whitegrid')
sns.set_context('talk')

df = pd.read_excel('Table S1.xlsx', sheet_name=0)

# Map BV status
bv_map = {'BV-P': 'BV Positive', 'BV-N': 'BV Negative', 'BV-T': 'Transitional BV'}
df['BV Status'] = df['BV Status'].map(bv_map)

# Map Gender
gender_map = {0: 'Female', 1: 'Male', 2: 'Unknown'}
df['Gender Label'] = df['Pt. Gender'].map(gender_map)

# hrHPV mapping
df['hrHPV Label'] = df['hrHPV Result'].map({0: 'Negative', 1: 'Positive'})

# Bacteria
bacterial_cols = [
    'F. vaginae ', 'G. vaginalis ', 'BVAB-2 ',
    'Megasphaera sp.Type 1 ', 'Megasphaera sp. Type 2 ',
    'L. iners ', 'L. gasseri ', 'L. jensenii ', 'L. crispatus '
]
bact_means, bact_pos_counts = [], []
for col in bacterial_cols:
    pos = df[col].fillna(0) > 0
    bact_means.append(df.loc[pos, col].mean())
    bact_pos_counts.append(pos.sum())

# HPV: mean concentration (pos only), count of positives for annotation
hpv_cols = [
    'HPV 16 ', 'HPV 18 ', 'HPV 31 ', 'HPV 33 ', 'HPV 35 ', 'HPV 39 ',
    'HPV 45 ', 'HPV 51 ', 'HPV 52 ', 'HPV 56 ', 'HPV 58 ', 'HPV 59 ', 'HPV 68 '
]
hpv_means, hpv_pos_counts = [], []
for col in hpv_cols:
    pos = df[col].fillna(0) > 0
    hpv_means.append(df.loc[pos, col].mean())
    hpv_pos_counts.append(pos.sum())

# BV status
bv_counts = df['BV Status'].value_counts().reindex(['BV Positive','BV Negative','Transitional BV']).fillna(0)
bv_total = bv_counts.sum()
bv_perc = [f"{int(x)} ({x/bv_total*100:.1f}%)" for x in bv_counts]

# hrHPV
hrhpv_counts = df['hrHPV Label'].value_counts().reindex(['Negative','Positive']).fillna(0)
hrhpv_total = hrhpv_counts.sum()
hrhpv_perc = [f"{int(x)} ({x/hrhpv_total*100:.1f}%)" for x in hrhpv_counts]

# Gender
gender_counts = df['Gender Label'].value_counts().reindex(['Female','Male','Unknown']).fillna(0)
gender_total = gender_counts.sum()
gender_perc = [f"{int(x)} ({x/gender_total*100:.1f}%)" for x in gender_counts]

# Cytology: only 1=positive
cyto_cols = ['NILM', 'ECA:ASCUS', 'ECA:ASC-H', 'ECA:LSIL', 'ECA:HSIL', 'ECA:AGC', 'RCC']
cyto_labels = ['NILM','ASCUS','ASC-H','LSIL','HSIL','AGC','RCC']
cyto_counts = [(df[col].fillna(0) == 1).sum() for col in cyto_cols]

# Age, Provider state
age_series = df['Pt. Age']
state_counts = df['Provider State'].value_counts().sort_values(ascending=False)

# Make a big figure, with more height and space between upper and lower panels
fig = plt.figure(figsize=(26, 18), dpi=400)
gs = gridspec.GridSpec(2, 4, width_ratios=[1.5, 1.5, 1, 2.2], height_ratios=[1, 1.1], wspace=0.36, hspace=0.6)

# A: Bacterial species means
ax1 = plt.subplot(gs[0, 0])
bars = ax1.bar([b.strip() for b in bacterial_cols], bact_means, color='#386cb0', edgecolor='black', linewidth=1.2)
ax1.set_title('A) Bacterial Species Mean', fontweight='bold')
ax1.set_ylabel('Mean Concentration')
ax1.set_xticklabels([b.strip() for b in bacterial_cols], rotation=45, ha='right', fontsize=14)
ax1.set_ylim(0, max(bact_means)*1.30)
for rect, count in zip(bars, bact_pos_counts):
    ax1.text(rect.get_x() + rect.get_width()/2., rect.get_height()+max(bact_means)*0.03, f'n={count}', ha='center', va='bottom', fontsize=12, rotation=90)

# B: HPV mean concentrations with positive count on bars
ax2 = plt.subplot(gs[0, 1])
bars = ax2.bar([h.strip() for h in hpv_cols], hpv_means, color='#fdb462', edgecolor='black', linewidth=1.2)
ax2.set_title('B) HPV Subtype Mean', fontweight='bold')
ax2.set_ylabel('Mean Concentration')
ax2.set_xticklabels([h.strip() for h in hpv_cols], rotation=45, ha='right', fontsize=14)
ax2.set_ylim(0, max(hpv_means)*1.25 if max(hpv_means)>0 else 1)
for rect, count in zip(bars, hpv_pos_counts):
    if not np.isnan(rect.get_height()) and rect.get_height() > 0:
        ax2.text(rect.get_x() + rect.get_width()/2., rect.get_height()+max(hpv_means)*0.03, f'n={count}', ha='center', va='bottom', fontsize=12, rotation=90)

# C: BV status with counts and percents
ax3 = plt.subplot(gs[0, 2])
bars = ax3.bar(bv_counts.index, bv_counts.values, color='#7fc97f', edgecolor='black', linewidth=1.2)
ax3.set_title('C) BV Status Distribution', fontweight='bold')
ax3.set_ylabel('Count')
ax3.set_xticklabels(bv_counts.index, rotation=45, ha='right', fontsize=14)
for idx, (rect, label) in enumerate(zip(bars, bv_perc)):
    # Shift BV Positive and BV Negative to the left edge of the bar
    if idx in [0, 1]:
        ax3.text(rect.get_x(), rect.get_height(), label, ha='left', va='bottom', fontsize=12)
    else:
        ax3.text(rect.get_x() + rect.get_width()/2., rect.get_height(), label, ha='center', va='bottom', fontsize=12)


# D: hrHPV result with counts and percents
ax4 = plt.subplot(gs[0, 3])
bars = ax4.bar(hrhpv_counts.index, hrhpv_counts.values, color='#ef3b2c', edgecolor='black', linewidth=1.2)
ax4.set_title('D) hrHPV Result Distribution', fontweight='bold')
ax4.set_ylabel('Count')
for rect, label in zip(bars, hrhpv_perc):
    ax4.text(rect.get_x() + rect.get_width()/2., rect.get_height(), label, ha='center', va='bottom', fontsize=12)

# E: Cytology results
ax5 = plt.subplot(gs[1, 0])
bars = ax5.bar(cyto_labels, cyto_counts, color='#beaed4', edgecolor='black', linewidth=1.2)
ax5.set_title('E) Pap Smear Cytology Results', fontweight='bold')
ax5.set_ylabel('Count')
ax5.set_xticklabels(cyto_labels, rotation=45, ha='right', fontsize=14)
for rect, count in zip(bars, cyto_counts):
    if count > 0:
        ax5.text(rect.get_x() + rect.get_width()/2., rect.get_height(), f'{int(count)}', ha='center', va='bottom', fontsize=12)

# F: Age distribution
ax6 = plt.subplot(gs[1, 1])
sns.histplot(age_series, bins=30, kde=True, color='#666666', ax=ax6)
ax6.set_title('F) Age Distribution', fontweight='bold')
ax6.set_xlabel('Age')
ax6.set_ylabel('Frequency')

# G: Gender distribution, with percent
ax7 = plt.subplot(gs[1, 2])
bars = ax7.bar(gender_counts.index, gender_counts.values, color='#fccde5', edgecolor='black', linewidth=1.2)
ax7.set_title('G) Gender Distribution', fontweight='bold')
ax7.set_ylabel('Count')
for idx, (rect, label) in enumerate(zip(bars, gender_perc)):
    # Female (index 0): shift annotation to right
    if idx == 0:
        ax7.text(rect.get_x() + rect.get_width() * 0.01, rect.get_height(), label,
                 ha='left', va='bottom', fontsize=12)
    else:
        # Rotate others at 90°
        ax7.text(rect.get_x() + rect.get_width()/2., rect.get_height(), label,
                 ha='center', va='bottom', fontsize=12, rotation=90)


# H: Provider State, annotations at 45°
ax8 = plt.subplot(gs[1, 3])
bars = ax8.bar(state_counts.index, state_counts.values, color='#b3b3b3', edgecolor='black', linewidth=1.2)
ax8.set_title('H) Provider Location Distribution', fontweight='bold', pad=35)
ax8.set_ylabel('Count')
ax8.set_xticklabels(state_counts.index, rotation=90, ha='center', fontsize=11)
for rect in bars:
    if rect.get_height() > 0:
        ax8.text(rect.get_x() + rect.get_width()/2., rect.get_height(), f'{int(rect.get_height())}', ha='center', va='bottom', fontsize=8, rotation=45)

plt.suptitle('Figure 1: Epidemiological & Microbiological Summary', fontsize=28, y=0.98, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Figure1_publication_ready.png', dpi=400)