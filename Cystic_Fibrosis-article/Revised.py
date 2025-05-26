import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# === Load variant decoding ===
variant_file = "Encoded_Dataset.xlsx"
sheet_map = {
    "Variant": "Variant_Codebook",
    "mRNA": "mRNA_Codebook",
    "Protein": "Protein_Codebook",
    "Common Name": "Common Name_Codebook"
}

decoders = {}
for key, sheet in sheet_map.items():
    df = pd.read_excel(variant_file, sheet_name=sheet)
    df.columns = ["Code", "Decoded"]
    decoders[key] = dict(zip(df["Code"].astype(str), df["Decoded"]))

# === Decode function ===
def simplify_label(label):
    base = label.split("__")[-1]
    parts = base.split("_")
    last = parts[-1]
    
    if "Variant" in base:
        return decoders["Variant"].get(last, last)
    elif "mRNA" in base:
        return decoders["mRNA"].get(last, last)
    elif "Protein" in base:
        return decoders["Protein"].get(last, last)
    elif "Common" in base:
        return decoders["Common Name"].get(last, last)
    else:
        return last.capitalize()

# === Load forest plot ===
forest_file = "LR_original_forestplot.csv"
df_forest = pd.read_csv(forest_file)
df_forest["Label"] = df_forest["Feature"].apply(simplify_label)

# === Load static images ===
imgs = {
    "conf_orig": mpimg.imread("Confusion_original.png"),
    "conf_filt": mpimg.imread("Confusion_filtered.png"),
    "roc_orig": mpimg.imread("ROC_original.png"),
    "roc_filt": mpimg.imread("ROC_filtered.png"),
    "shap": mpimg.imread("SHAP_filtered.png")
}

# === Create figure ===
fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=300)

# --- Top Row ---
axes[0, 0].imshow(imgs["conf_orig"])
axes[0, 0].axis("off")
axes[0, 0].set_title("Confusion Matrix – Original", fontsize=10)

axes[0, 1].imshow(imgs["conf_filt"])
axes[0, 1].axis("off")
axes[0, 1].set_title("Confusion Matrix – Filtered", fontsize=10)

axes[0, 2].imshow(imgs["roc_orig"])
axes[0, 2].axis("off")
axes[0, 2].set_title("ROC Curve – Original", fontsize=10)

# --- Bottom Row ---

from matplotlib.ticker import LogLocator, LogFormatter

# Forest Plot with cleaned formatting
ax_fp = axes[1, 0]
ax_fp.barh(df_forest["Label"], df_forest["OddsRatio"], color="teal")

# Set log scale and formatter
ax_fp.set_xscale("log")
ax_fp.xaxis.set_major_locator(LogLocator(base=10.0))  # control spacing
ax_fp.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=True))  # clearer ticks

# Titles and labels
ax_fp.set_title("Logistic Regression Odds Ratios", fontsize=10)
ax_fp.set_xlabel("Odds Ratio (log scale)", fontsize=8)

# Clean font and rotation
ax_fp.tick_params(axis="x", labelsize=7, rotation=45)
ax_fp.tick_params(axis="y", labelsize=6)

# SHAP
axes[1, 1].imshow(imgs["shap"])
axes[1, 1].axis("off")
axes[1, 1].set_title("SHAP Summary – Filtered", fontsize=10)

# ROC – Filtered
axes[1, 2].imshow(imgs["roc_filt"])
axes[1, 2].axis("off")
axes[1, 2].set_title("ROC Curve – Filtered", fontsize=10)

# === Save output ===
plt.subplots_adjust(bottom=0.15)  # Give space for rotated labels
plt.tight_layout()
plt.savefig("Sensitivity_Composite_Final_Cleaned_300dpi.png", dpi=300)
plt.show()
