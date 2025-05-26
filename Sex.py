import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal, chi2_contingency

# === Load data ===
df = pd.read_excel("Empty_variant_rows_removed.xlsx")

# === Boxplot: Mutation count per patient by sex ===
mut_count = df.groupby(["Pt Gender", "MDLNo"]).size().reset_index(name="Mutation Count")

plt.figure(figsize=(8, 6))
sns.boxplot(data=mut_count, x="Pt Gender", y="Mutation Count", palette="Set2")
sns.stripplot(data=mut_count, x="Pt Gender", y="Mutation Count", color="black", alpha=0.5)
plt.title("Mutation Count per Patient by Sex")
plt.tight_layout()
plt.savefig("Boxplot_Mutations_By_Sex.png", dpi=300)
plt.show()

# === Kruskal-Wallis Test ===
groups = [g["Mutation Count"].values for name, g in mut_count.groupby("Pt Gender")]
stat, p = kruskal(*groups)
print(f"Kruskal-Wallis p-value = {p:.4f}")

# === Stacked Barplot: Variant Frequency by Sex ===
variant_freq = df.groupby(["Pt Gender", "Variant"]).size().reset_index(name="Count")
variant_top10 = variant_freq.groupby("Variant")["Count"].sum().nlargest(10).index.tolist()
variant_freq = variant_freq[variant_freq["Variant"].isin(variant_top10)]

variant_pivot = variant_freq.pivot(index="Variant", columns="Pt Gender", values="Count").fillna(0)

# Chi-square test
chi2, p2, _, _ = chi2_contingency(variant_pivot.T)
print(f"Chi-square p-value for top variants by sex = {p2:.4f}")

# Plot
variant_pivot.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Accent")
plt.title("Top Variant Frequencies by Sex")
plt.ylabel("Frequency")
plt.xlabel("Variant")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("StackedBar_TopVariants_By_Sex.png", dpi=300)
plt.show()
