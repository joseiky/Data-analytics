import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ===== 1. Load Main Clinical Dataset =====
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

# ===== 2. Load Variant Table (No Empty Rows) =====
df_variant_raw = pd.read_excel("Empty_variant_rows_removed.xlsx")

# ===== 3. Decode Codes =====
cb = "Encoded_Dataset.xlsx"
variant_dict = pd.read_excel(cb, sheet_name="Variant_Codebook").set_index("Code")["Label"].to_dict()
mRNA_dict = pd.read_excel(cb, sheet_name="mRNA_Codebook").set_index("Code")["Label"].to_dict()
protein_dict = pd.read_excel(cb, sheet_name="Protein_Codebook").set_index("Code")["Label"].to_dict()
common_dict = pd.read_excel(cb, sheet_name="Common Name_Codebook").set_index("Code")["Label"].to_dict()

df_variant_raw["Variant_decoded"] = df_variant_raw["Variant"].astype(str).str.strip().map(variant_dict)
df_variant_raw["mRNA_decoded"] = df_variant_raw["mRNA"].astype(str).str.strip().map(mRNA_dict)
df_variant_raw["Protein_decoded"] = df_variant_raw["Protein"].astype(str).str.strip().map(protein_dict)
df_variant_raw["Common Name_decoded"] = df_variant_raw["Common Name"].astype(str).str.strip().map(common_dict)

# ===== 4. Styling =====
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7
})

# ===== 5. Plotting Utilities =====
def chi2_p(a, b):
    try:
        return chi2_contingency(pd.crosstab(a, b))[1]
    except:
        return None

def p_to_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def plot_sourcewise(ax, data, outcome, rotation=90, top_n=None):
    df_plot = data[["Source", outcome]].dropna()
    if top_n:
        top_vals = df_plot[outcome].value_counts().nlargest(top_n).index
        df_plot = df_plot[df_plot[outcome].isin(top_vals)]
    if df_plot.empty:
        ax.axis("off")
        return
    sns.countplot(data=df_plot, x="Source", hue=outcome, ax=ax,
                  order=df_plot["Source"].value_counts().index)
    ax.tick_params(axis='x', rotation=rotation)
    for container in ax.containers:
        ax.bar_label(container, fontsize=6, padding=2)
    p = chi2_p(df_plot["Source"], df_plot[outcome])
    ax.set_title(f"{outcome} by Specimen Source ({p_to_stars(p)})", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(loc='best', fontsize=6, title=outcome)

# ===== 6. Define Variables =====
demo_vars = ["Pt Gender", "Age Group", "Ethnicity", "Zygosity", "Result", "Abnormal"]
variant_cols = {
    "Variant_decoded": "Variant_decoded",
    "mRNA_decoded": "mRNA_decoded",
    "Protein_decoded": "Protein_decoded",
    "Common Name_decoded": "Common Name_decoded"
}

# ===== 7. Plot All Subplots =====
fig, axs = plt.subplots(4, 3, figsize=(18, 22))
axs = axs.flatten()

# Demographic plots
for i, var in enumerate(demo_vars):
    plot_sourcewise(axs[i], df, var, rotation=90)

# Variant plots: show top 10 only
for j, (label, colname) in enumerate(variant_cols.items()):
    df_var_filtered = df_variant_raw[df_variant_raw[colname].notna()]
    plot_sourcewise(axs[len(demo_vars) + j], df_var_filtered, colname, rotation=90, top_n=10)

# Hide final unused subplot
for k in range(len(demo_vars) + len(variant_cols), len(axs)):
    axs[k].axis("off")

plt.tight_layout()

# ===== 8. Save Output =====
plt.savefig("Figure_1_SpecimenSource_11plots_TOP10.png", dpi=500)
plt.savefig("Figure_1_SpecimenSource_11plots_TOP10.tiff", dpi=300)
plt.show()
