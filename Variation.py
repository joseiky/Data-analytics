import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === LOAD DATA ===
df = pd.read_excel("Longitudinal_Patient_Variation_Data.xlsx")
df["Date"] = pd.to_datetime(df["Date Specimen Collected"], errors="coerce")
df["YM"] = df["Date"].dt.to_period("M").dt.to_timestamp()

# === DECODE PROTEIN ===
protein_map = pd.read_excel("Encoded_Dataset.xlsx", sheet_name="Protein_Codebook")
protein_map.columns = ["code", "label"]
protein_dict = dict(zip(protein_map["code"].astype(str), protein_map["label"]))
df["Protein Decoded"] = df["Protein"].astype(str).map(protein_dict)

# === FILTER FOR PATIENTS WITH VARIATIONS ===
variation_counts = (
    df.groupby("MDL Patient ID")["Protein Decoded"]
    .nunique()
    .reset_index(name="Unique Variants")
)
varied_patients = variation_counts[variation_counts["Unique Variants"] > 1]["MDL Patient ID"]
df = df[df["MDL Patient ID"].isin(varied_patients)].copy()

# === COMBINE SAMPLE SOURCE & TYPE ===
df["Sample_Y"] = df["Specimen"].astype(str) + " - " + df["Source"].astype(str)

# === HANDLE ABNORMAL & ZYGOSITY ===
df["Zygosity"] = df["Zygosity"].fillna("Unknown")
df["Abnormal Flag"] = df["Abnormal Flag"].fillna("N")
df["Bar ID"] = df["Protein Decoded"] + " | " + df["Zygosity"] + " | " + df["Abnormal Flag"]

# === PLOT 1: HORIZONTAL STACKED BAR (ALL PATIENTS TOGETHER) ===
plt.figure(figsize=(18, 8))
ax1 = sns.stripplot(
    data=df,
    x="YM",
    y="Sample_Y",
    hue="Bar ID",
    dodge=True,
    size=8,
    jitter=False,
    palette="tab20"
)
ax1.tick_params(axis='x', rotation=45)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xlabel("Collection Date (Year-Month)")
plt.ylabel("Sample Source and Type")
plt.title("Protein Variations Across All Patients by Source and Collection Date")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Protein | Zygosity | Abnormal")
plt.tight_layout()
plt.savefig("Protein_Variation_HorizontalBar_AllPatients.png", dpi=600)
plt.show()

# === PLOT 2: SAME DATA, GROUPED BY PATIENT ===
g = sns.FacetGrid(
    df,
    row="MDL Patient ID",
    height=3,
    aspect=3,
    sharey=False
)
g.map_dataframe(
    sns.stripplot,
    x="YM",
    y="Sample_Y",
    hue="Bar ID",
    dodge=True,
    size=6,
    jitter=False,
    palette="tab20"
)
g.set_axis_labels("Collection Date (Year-Month)", "Sample Source and Type")
g.set_titles("Patient ID: {row_name}")
for ax in g.axes.flatten():
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
g.add_legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Protein | Zygosity | Abnormal")
plt.subplots_adjust(top=0.95)
g.fig.suptitle("Protein Variations Grouped by Patient and Sample Collection")
g.savefig("Protein_Variation_ByPatient_Faceted.png", dpi=600)
plt.show()
