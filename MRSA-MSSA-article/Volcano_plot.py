
+import pandas as pd
+import numpy as np
+import matplotlib.pyplot as plt
+import seaborn as sns
+from pathlib import Path
+from scipy.stats import chi2_contingency, fisher_exact
+from statsmodels.stats.multitest import multipletests
+
+OUT = Path("reanalysed_outputs")
+OUT.mkdir(exist_ok=True)
+
+# Load cleaned dataset
+DF_PATH = Path("MRSA-MSSA-article") / "Table S1. Dataset-MRSA-MSSA.xlsx"
+df = pd.read_excel(DF_PATH, sheet_name="1.0 Cleaned_Data")
+
+# List of cytology outcomes (adjust if dataset uses a different set)
+cytology_cols = [
+    "NILM", "ECA", "AGC", "ASC-H", "ASCUS", "RCC", "LSIL", "HSIL",
+    # add additional columns if present in the dataset
+]
+
+results = []
+for col in cytology_cols:
+    pos = df[col] == col
+    table = pd.crosstab(df["BV-Status"] == "BV-POSITIVE", pos)
+    for flag in [False, True]:
+        if flag not in table.columns:
+            table[flag] = 0
+    table = table[[False, True]]
+    chi2, p, _, exp = chi2_contingency(table, correction=False)
+    if (exp < 5).any():
+        orr, p = fisher_exact(table)
+    else:
+        orr = (table.loc[True, True] * table.loc[False, False]) / (
+            (table.loc[False, True] + 1e-9) * (table.loc[True, False] + 1e-9)
+        )
+    results.append(dict(Outcome=col, OR=orr, p=p))
+
+pvals = [r["p"] for r in results]
+padj = multipletests(pvals, method="fdr_bh")[1]
+for r, pa in zip(results, padj):
+    r["p_adj"] = pa
+
+res = pd.DataFrame(results)
+res["log10_OR"] = np.log10(res["OR"])
+res["neglog10_padj"] = -np.log10(res["p_adj"])
+
+plt.figure(figsize=(6, 4))
+sns.scatterplot(x="log10_OR", y="neglog10_padj", data=res)
+for _, row in res.iterrows():
+    plt.text(row["log10_OR"], row["neglog10_padj"], row["Outcome"], fontsize=8)
+plt.axvline(0, color="gray", linestyle="--")
+plt.axhline(-np.log10(0.05), color="red", linestyle="--")
+plt.xlabel("log10(OR)")
+plt.ylabel("-log10(FDR p)")
+plt.title("Cytology outcomes volcano")
+plt.tight_layout()
+plt.savefig(OUT / "fig_SUPP_cytology_volcano.png", dpi=300)
