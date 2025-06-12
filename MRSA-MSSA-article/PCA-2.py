import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load plotting data
within_df = pd.read_csv('Within_Patient_plotting_data.csv')

# ========== 1. Within-patient S. aureus Mean and SD ==========
saureus_data = within_df[['Patient-ID', 'S_aureus_conc']].dropna()
grouped = saureus_data.groupby('Patient-ID')['S_aureus_conc']
within_means = grouped.mean()
within_sds = grouped.std()

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(within_means, kde=True, ax=axs[0], color='blue')
axs[0].set_title("Distribution of Within-Patient Mean S. aureus Conc.")
axs[0].set_xlabel("Mean S. aureus Concentration")
axs[0].set_ylabel("Patient Count")
sns.histplot(within_sds.dropna(), kde=True, ax=axs[1], color='orange')
axs[1].set_title("Distribution of Within-Patient SD (S. aureus Conc.)")
axs[1].set_xlabel("SD of S. aureus Concentration")
axs[1].set_ylabel("Patient Count")
plt.tight_layout()
plt.savefig('WithinPatient_Saureus_MeanSD.png', dpi=400)
plt.show()

# ========== 2. BV-Positive and BV-Transitional: Proportions ==========
bv_df = within_df[['Patient-ID', 'BV_P', 'BV_T']].dropna()
prop_BV_P = bv_df.groupby('Patient-ID')['BV_P'].mean()
prop_BV_T = bv_df.groupby('Patient-ID')['BV_T'].mean()

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
sns.histplot(prop_BV_P, kde=True, ax=axs[0], color='purple')
axs[0].set_title("Within-Patient Proportion: BV-Positive")
axs[0].set_xlabel("Proportion BV-Positive")
axs[0].set_ylabel("Patient Count")
sns.histplot(prop_BV_T, kde=True, ax=axs[1], color='green')
axs[1].set_title("Within-Patient Proportion: BV-Transitional")
axs[1].set_xlabel("Proportion BV-Transitional")
axs[1].set_ylabel("Patient Count")
plt.tight_layout()
plt.savefig('WithinPatient_BV_Proportions.png', dpi=400)
plt.show()

# ========== 3. Violin Plot of BV State Proportions ==========
bv_means_long = pd.melt(pd.DataFrame({
    'BV-Positive': prop_BV_P,
    'BV-Transitional': prop_BV_T
}), var_name='BV State', value_name='Proportion')

plt.figure(figsize=(7, 5))
sns.violinplot(x='BV State', y='Proportion', data=bv_means_long, inner='box')
plt.title("Distribution of Within-Patient Proportion of BV States")
plt.ylabel("Proportion of Samples")
plt.tight_layout()
plt.savefig('WithinPatient_BV_Violin.png', dpi=400)
plt.show()
