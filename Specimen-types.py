import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ========== 1. Load Your Data ==========
df_demo = pd.read_excel("Dataset.xlsx")
variant_xlsx = pd.ExcelFile("Specimen-variants_Tables.xlsx")
df_variant = pd.concat(
    [pd.read_excel(variant_xlsx, sheet_name=s).assign(Specimen=s) for s in variant_xlsx.sheet_names],
    ignore_index=True
)

# ========== 2. Decode Variant Columns ==========
cb = "Encoded_Dataset.xlsx"
variant_dict = pd.read_excel(cb, sheet_name="Variant_Codebook").set_index("Code")["Label"].to_dict()
mRNA_dict = pd.read_excel(cb, sheet_name="mRNA_Codebook").set_index("Code")["Label"].to_dict()
protein_dict = pd.read_excel(cb, sheet_name="Protein_Codebook").set_index("Code")["Label"].to_dict()
common_dict = pd.read_excel(cb, sheet_name="Common Name_Codebook").set_index("Code")["Label"].to_dict()

df_variant["Variant_decoded"] = df_variant["Variant"].astype(str).map(variant_dict)
df_variant["mRNA_decoded"] = df_variant["mRNA"].astype(str).map(mRNA_dict)
df_variant["Protein_decoded"] = df_variant["Protein"].astype(str).map(protein_dict)
df_variant["Common Name_decoded"] = df_variant["Common Name"].astype(str).map(common_dict)

# ========== 3. Preprocess Demo Data ==========
df_demo["Abnormal"] = df_demo.iloc[:, 17].map({"A": "Abnormal", "N": "Normal"})
df_demo["Zygosity"] = df_demo.iloc[:, 15]
df_demo["Result"] = df_demo.iloc[:, 16]
df_demo["Source"] = df_demo.iloc[:, 2]
df_demo["Specimen"] = df_demo.iloc[:, 1]
df_demo["Pt Gender"] = df_demo["Pt Gender"]
df_demo["Ethnicity"] = df_demo["Ethnicity"]
df_demo = df_demo[df_demo["Zygosity"].isin(["Heterozygous", "Homozygous"])]
df_demo["Age Group"] = pd.cut(
    df_demo["AGE"],
    bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 200],
    labels=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
)

# ========== 4. Helper Functions ==========
def chi2_p(a, b):
    try:
        return chi2_contingency(pd.crosstab(a, b))[1]
    except:
        return None

def p_to_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def polished_barplot(ax, data, x, hue="Specimen"):
    data = data[[x, hue]].dropna()
    if data.empty:
        ax.axis("off")
        return
    sns.countplot(data=data, x=x, hue=hue, ax=ax, order=data[x].value_counts().index)
    ax.tick_params(axis='x', rotation=45)
    for container in ax.containers:
        ax.bar_label(container, fontsize=6, padding=2)
    p = chi2_p(data[x], data[hue])
    if p is not None:
        ax.set_title(f"{x} vs Specimen ({p_to_stars(p)})", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Count")

# ========== 5. Plotting Config ==========
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.titlesize": 14
})

# ========== 6. Generate Subplots (with smarter label rotation) ==========
fig, axs = plt.subplots(4, 3, figsize=(18, 22))
axs = axs.flatten()

# First 7 from Dataset.xlsx
demo_vars = ["Source", "Pt Gender", "Age Group", "Ethnicity", "Zygosity", "Result", "Abnormal"]
for i, var in enumerate(demo_vars):
    data = df_demo[[var, "Specimen"]].dropna()
    sns.countplot(data=data, x=var, hue="Specimen", ax=axs[i], order=data[var].value_counts().index)
    axs[i].tick_params(axis='x', rotation=45)
    for c in axs[i].containers:
        axs[i].bar_label(c, fontsize=6, padding=2)
    p = chi2_p(data[var], data["Specimen"])
    axs[i].set_title(f"{var} vs Specimen ({p_to_stars(p)})", fontsize=11)
    axs[i].set_xlabel("")
    axs[i].set_ylabel("Count")

# Last 4 variant plots (vertical x-axis labels)
variant_vars = ["Variant_decoded", "mRNA_decoded", "Protein_decoded", "Common Name_decoded"]
for j, var in enumerate(variant_vars):
    idx = len(demo_vars) + j
    data = df_variant[[var, "Specimen"]].dropna()
    sns.countplot(data=data, x=var, hue="Specimen", ax=axs[idx], order=data[var].value_counts().index)
    axs[idx].tick_params(axis='x', rotation=90)
    for c in axs[idx].containers:
        axs[idx].bar_label(c, fontsize=6, padding=2)
    p = chi2_p(data[var], data["Specimen"])
    axs[idx].set_title(f"{var} vs Specimen ({p_to_stars(p)})", fontsize=11)
    axs[idx].set_xlabel("")
    axs[idx].set_ylabel("Count")

# Hide any extra panels
for k in range(len(demo_vars) + len(variant_vars), len(axs)):
    axs[k].axis("off")

# ========== 7. Export ==========
plt.tight_layout()
plt.savefig("Figure_11_Subplots_Beautiful_Adjusted.png", dpi=500)
plt.savefig("Figure_11_Subplots_Beautiful_Adjusted.tiff", dpi=500)
plt.show()
