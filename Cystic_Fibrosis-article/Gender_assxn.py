import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ===== 1. Load Clinical Dataset =====
df = pd.read_excel("Dataset.xlsx")
df["Source"] = df.iloc[:, 2]
df["Specimen"] = df.iloc[:, 1]
df["Pt Gender"] = df["Pt Gender"]
df["Ethnicity"] = df["Ethnicity"]
df["Zygosity"] = df.iloc[:, 15]
df["Result"] = df.iloc[:, 16]
df["Abnormal"] = df.iloc[:, 17].map({"A": "Abnormal", "N": "Normal"})
df["Age Group"] = pd.cut(df["AGE"], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 200],
                         labels=['0-9', '10-19', '20-29', '30-39', '40-49',
                                 '50-59', '60-69', '70-79', '80+'])

# ===== 2. Load Variant Table and Decode =====
df_variant = pd.read_excel("Empty_variant_rows_removed.xlsx")

cb = "Encoded_Dataset.xlsx"
variant_dict = pd.read_excel(cb, sheet_name="Variant_Codebook").set_index("Code")["Label"].to_dict()
mRNA_dict = pd.read_excel(cb, sheet_name="mRNA_Codebook").set_index("Code")["Label"].to_dict()
protein_dict = pd.read_excel(cb, sheet_name="Protein_Codebook").set_index("Code")["Label"].to_dict()
common_dict = pd.read_excel(cb, sheet_name="Common Name_Codebook").set_index("Code")["Label"].to_dict()

df_variant["Variant_decoded"] = df_variant["Variant"].astype(str).str.strip().map(variant_dict)
df_variant["mRNA_decoded"] = df_variant["mRNA"].astype(str).str.strip().map(mRNA_dict)
df_variant["Protein_decoded"] = df_variant["Protein"].astype(str).str.strip().map(protein_dict)
df_variant["Common Name_decoded"] = df_variant["Common Name"].astype(str).str.strip().map(common_dict)

# ===== 3. Plot Styling =====
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7
})

# ===== 4. Utility Functions =====
def chi2_p(a, b):
    try:
        return chi2_contingency(pd.crosstab(a, b))[1]
    except:
        return None

def p_to_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def plot_genderwise(ax, data, x, rotation=90, top_n=None):
    df_plot = data[["Pt Gender", x]].dropna()
    if top_n:
        top_vals = df_plot[x].value_counts().nlargest(top_n).index
        df_plot = df_plot[df_plot[x].isin(top_vals)]
    if df_plot.empty:
        ax.axis("off")
        return
    sns.countplot(data=df_plot, x=x, hue="Pt Gender", ax=ax, order=df_plot[x].value_counts().index)
    ax.tick_params(axis='x', rotation=rotation)
    for container in ax.containers:
        ax.bar_label(container, fontsize=6, padding=2)
    p = chi2_p(df_plot["Pt Gender"], df_plot[x])
    ax.set_title(f"Gender vs {x} ({p_to_stars(p)})", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(loc='best', fontsize=6)

# ===== 5. Variables to Plot =====
demo_vars = ["Source", "Specimen", "Age Group", "Ethnicity", "Result", "Abnormal"]
variant_vars = ["Variant_decoded", "mRNA_decoded", "Protein_decoded", "Common Name_decoded"]

# ===== 6. Plotting =====
fig, axs = plt.subplots(4, 3, figsize=(18, 22))
axs = axs.flatten()

# First 6 demographic plots
for i, var in enumerate(demo_vars):
    plot_genderwise(axs[i], df, var, rotation=90)

# Last 4 variant plots (top 10 only)
for j, var in enumerate(variant_vars):
    plot_genderwise(axs[len(demo_vars) + j], df_variant, var, rotation=90, top_n=10)

# Hide any unused subplots
for k in range(len(demo_vars) + len(variant_vars), len(axs)):
    axs[k].axis("off")

plt.tight_layout()

# ===== 7. Save Outputs =====
plt.savefig("Figure_3_Gender_vs_Others.png", dpi=500)
plt.savefig("Figure_3_Gender_vs_Others.tiff", dpi=400)
plt.show()
