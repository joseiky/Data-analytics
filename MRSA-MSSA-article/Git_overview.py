import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
df = pd.read_excel("GIT-No-toxins.xlsx")

# Standardize column names
df.columns = df.columns.str.strip().str.replace(r"[\s\-]+", "_", regex=True).str.lower()

# Replace \N, blanks, and drop all nulls in analysis columns
df.replace(r"^\s*\\?N\s*$", pd.NA, regex=True, inplace=True)
df = df.dropna(subset=['specimen', 'source', 'bv_status', 'gender', 'ethnicity', 'age'])

# Clean and convert age to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df.dropna(subset=['age'])

# Bin age into 10-year intervals (0–9, 10–19, ..., 110–119)
age_bins = list(range(0, 121, 10))  # 0, 10, ..., 120
age_labels = [f"{a}-{b-1}" for a, b in zip(age_bins[:-1], age_bins[1:])]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# Define columns to analyze and plot
columns = {
    "specimen": "Top 10 Specimen Types",
    "source": "Top 10 Sample Sources",
    "bv_status": "BV Status",
    "gender": "Gender",
    "ethnicity": "Ethnicity",
    "age_group": "Age Group Distribution"
}

# Set style and create figure
sns.set(style="whitegrid")
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
axs = axs.flatten()

# Plot each subplot
for i, (col, title) in enumerate(columns.items()):
    data = df[col].dropna()
    if col in ['specimen', 'source']:
        data = data.value_counts().nlargest(10).reset_index()
    else:
        data = data.value_counts().reset_index()
    data.columns = [col, 'count']

    sns.barplot(x=col, y='count', data=data, ax=axs[i], palette="viridis")
    axs[i].set_title(title, fontsize=14)
    axs[i].tick_params(axis='x', rotation=90)
    axs[i].set_ylabel("Count")

    for j, val in enumerate(data['count']):
        axs[i].text(j, val + max(data['count']) * 0.01, str(int(val)),
                    ha='center', va='bottom', fontsize=8)

# Final layout
fig.suptitle("Figure 1: GIT Dataset Overview – Sampling and Demographics", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("Figure1_GIT_Corrected_Final.png", dpi=400)

print("✅ Saved: Figure1_GIT_Corrected_Final.png")
