import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ===== 1. Load Dataset.xlsx =====
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

# ===== 2. Load Variant Table =====
df_variant = pd.read_excel("Empty_variant_rows_removed.xlsx")

# ===== 3. Decode variant values =====
cb = "Encoded_Dataset.xlsx"
variant_dict = pd.read_excel(cb, sheet_name="Variant_Codebook").set_index("Code")["Label"].to_dict()
mRNA_dict = pd.read_excel(cb, sheet_name="mRNA_Codebook").set_index("Code")["Label"].to_dict()
protein_dict = pd.read_excel(cb, sheet_name="Protein_Codebook").set_index("Code")["Label"].to_dict()
common_dict = pd.read_excel(cb, sheet_name="Common Name_Codebook").set_index("Code")["Label"].to_dict()

df_variant["Variant_decoded"] = df_variant["Variant"].astype(str).str.strip().map(variant_dict)
df_variant["mRNA_decoded"] = df_variant["mRNA"].astype(str).str.strip().map(mRNA_dict)
df_variant["Protein_decoded"] = df_variant["Protein"].astype(str).str.strip().map(protein_dict)
df_variant["Common Name_decoded"] = df_variant["Common Name"].astype(str).str.strip().map(common_dict)

# ===== 4. Plot styling =====
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7
})

# ===== 5. Plot utilities =====
def chi2_p(a, b):
    try:
        return chi2_contingency(pd.crosstab(a, b))[1]
    except:
        return None

def p_to_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def plot_ethnicity_x(ax, data, hue, rotation=90, top_n=None):
    df_plot = data[["Ethnicity", hue]].dropna()
    if top_n:
        top_vals = df_plot[hue].value_counts().nlargest(top_n).index
        df_plot = df_plot[df_plot[hue].isin(top_vals)]
    if df_plot.empty:
        ax.axis("off")
        return
    sns.countplot(data=df_plot, x="Ethnicity", hue=hue, ax=ax, order=df_plot["Ethnicity"].value_counts().index)
    ax.tick_params(axis='x', rotation=rotation)
    for c in ax.containers:
        ax.bar_label(c, fontsize=6, padding=2)
    p = chi2_p(df_plot["Ethnicity"], df_plot[hue])
    ax.set_title(f"{hue} by Ethnicity ({p_to_stars(p)})", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(loc='best', fontsize=6)

# ===== 6. Variables =====
demo_hues = ["Source", "Specimen", "Age Group", "Pt Gender", "Result", "Abnormal"]
variant_hues = ["Variant_decoded", "mRNA_decoded", "Protein_decoded", "Common Name_decoded"]

# ===== 7. Create 10-subplot canvas =====
import matplotlib.pyplot as plt
fig, axs = plt.subplots(4, 3, figsize=(18, 22))
axs = axs.flatten()

for i, var in enumerate(demo_hues):
    plot_ethnicity_x(axs[i], df, var, rotation=90)

for j, var in enumerate(variant_hues):
    plot_ethnicity_x(axs[len(demo_hues) + j], df_variant, var, rotation=90, top_n=10)

# Hide any unused
for k in range(len(demo_hues) + len(variant_hues), len(axs)):
    axs[k].axis("off")

plt.tight_layout()

# ===== 8. Export =====
plt.savefig("Figure_4B_Ethnicity_on_X_Axis.png", dpi=300)
plt.savefig("Figure_4B_Ethnicity_on_X_Axis.tiff", dpi=300)
plt.show()
