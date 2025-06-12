import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Load data
df = pd.read_csv('Dataset_1_Cleaned_masterfile.csv', dtype=str)
df['Date-Collected'] = pd.to_datetime(df['Date-Collected'], errors='coerce')
df = df[df['Date-Collected'].notna()].copy()
df['Month'] = df['Date-Collected'].dt.to_period("M").dt.to_timestamp()

# List of pathologies
pathology_cols = [
    'ECA', 'LSIL', 'HSIL', 'ASCUS', 'AGC', 'RCC', 'ASC-H', 'CD', 'TRICH', 'Actino', 'Atrophy', 'BV', 'Atrophic VAG'
]

# Set months range for X axis
months = pd.date_range(df['Month'].min(), df['Month'].max(), freq='MS')

# Collect plotting data
plotting_data = []
for col in pathology_cols:
    path_df = df[df[col] == '1']
    monthly_counts = path_df.groupby('Month').size()
    monthly_counts = monthly_counts.reindex(months, fill_value=0)
    for m, cnt in zip(months, monthly_counts):
        plotting_data.append({'Pathology': col, 'Month': m, 'Count': cnt})

# Export plotting data
plot_df = pd.DataFrame(plotting_data)
plot_df.to_csv('FigureS9_MonthlyPathologyCounts.csv', index=False)

# PLOTTING (Publication-Ready)
sns.set_theme(style="whitegrid", font_scale=1.3)
n_panels = len(pathology_cols)
n_cols = 4
n_rows = int(np.ceil(n_panels / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5.5, n_rows*3), dpi=300)
axes = axes.flatten()

for idx, col in enumerate(pathology_cols):
    ax = axes[idx]
    # Get plotting data
    subset = plot_df[plot_df['Pathology'] == col]
    y = subset['Count'].values
    x = subset['Month'].dt.strftime('%Y-%m')
    bars = ax.bar(x, y, color="#5bc0eb", edgecolor='k', width=0.6)
    ax.set_title(f"{col} (Positive) Monthly Trend", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(bottom=0)
    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width()/2, height), 
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    # Linear trend for p-value
    x_idx = np.arange(len(x))
    if y.sum() > 0 and len(np.unique(y)) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(x_idx, y)
        if p_value < 0.001:
            p_annot = '*** (p<0.001)'
        elif p_value < 0.01:
            p_annot = '** (p<0.01)'
        elif p_value < 0.05:
            p_annot = '* (p<0.05)'
        else:
            p_annot = 'ns (p≥0.05)'
        ax.text(0.97, 0.93, p_annot, transform=ax.transAxes, ha='right', va='top', color='red', fontsize=12)
    else:
        ax.text(0.97, 0.93, 'No positive cases', transform=ax.transAxes, ha='right', va='top', color='gray', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Remove unused axes (if any)
for idx in range(len(pathology_cols), len(axes)):
    fig.delaxes(axes[idx])

fig.suptitle('Monthly Temporal Trends in Pathology Outcomes', fontsize=20, fontweight='bold', y=1.04)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Figure_S9_Pathology_Temporal_Trends_pubready.png", dpi=300)
plt.show()
