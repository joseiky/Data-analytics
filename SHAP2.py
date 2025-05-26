import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

# -------------------- 1. Load data --------------------
df = pd.read_excel("Empty_variant_rows_removed.xlsx")
sheets = pd.read_excel("Encoded_Dataset.xlsx", sheet_name=None)

# -------------------- 2. Build decode map --------------------
# sheets[1]..sheets[4] each have columns "Code" and "Label"
decode_map = {}
for sheet_name in list(sheets.keys())[1:5]:
    sub = sheets[sheet_name][['Code','Label']].dropna()
    decode_map.update(dict(zip(sub['Code'], sub['Label'])))

# -------------------- 3. Target setup --------------------
df = df.dropna(subset=["Abnormal Flag","Variant"])
df["Abnormal_Flag_Target_cl"] = df["Abnormal Flag"].map({"A":1,"N":0})
df = df.dropna(subset=["Abnormal_Flag_Target_cl"])
drop_cols = ["Abnormal Flag","Abnormal_Flag_Target_cl",
             "Date Specimen Collected","MDLNo","MDL Patient ID"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["Abnormal_Flag_Target_cl"]
X = X.fillna(0)

# -------------------- 4. One‑hot encode & align --------------------
X_full = pd.get_dummies(X).fillna(0).astype(np.float64)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.30, random_state=42, stratify=y
)
X_train = X_train_full.copy()
X_test  = X_test_full.reindex(columns=X_train.columns, fill_value=0)
X_train = X_train.astype(np.float64)
X_test  = X_test.astype(np.float64)

# -------------------- 5. Train model --------------------
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# -------------------- 6. SHAP values --------------------
explainer   = shap.TreeExplainer(rf, X_train, model_output="probability")
shap_values = explainer.shap_values(X_test)

# get the “Abnormal” class slice
if isinstance(shap_values, list):
    shap_matrix = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_matrix = shap_values[:, :, 1]
else:
    shap_matrix = shap_values

# -------------------- 7. Decode feature‑names --------------------
# For each one‑hot column like "mRNA_CODE123", pull out CODE123 → lookup → LABEL
decoded_cols = []
for col in X_test.columns:
    if "_" in col:
        prefix, code = col.split("_", 1)
        label = decode_map.get(code, code)
    else:
        label = col
    decoded_cols.append(label)

X_test_dec = X_test.copy()
X_test_dec.columns = decoded_cols

# sanity check
if shap_matrix.shape != X_test_dec.shape:
    print(f"\n❌ SHAP shape {shap_matrix.shape} != X_test {X_test_dec.shape}")
    sys.exit("Mismatch: check decode logic or SHAP API")

# -------------------- 8. Group by original variable --------------------
groups = {
    "Variant":     "Variant_",
    "mRNA":        "mRNA_",
    "Protein":     "Protein_",
    "Common Name": "Common Name_"
}
group_indices = {
    title: [i for i,c in enumerate(X_test.columns) if c.startswith(pref)]
    for title,pref in groups.items()
}

# -------------------- 9. Plot 2×2 grid --------------------
fig, axes = plt.subplots(2, 2, figsize=(16,12))
axes = axes.flatten()

for ax, (title, idx) in zip(axes, group_indices.items()):
    if not idx:
        ax.set_title(f"{title} (no features found)")
        continue

    # mean absolute SHAP per feature
    vals = np.abs(shap_matrix[:, idx]).mean(axis=0)
    top_n = min(10, len(idx))
    order = np.argsort(vals)[::-1][:top_n]
    y_pos = np.arange(top_n)

    top_vals   = vals[order]
    top_labels = np.array(X_test_dec.columns)[idx][order]

    ax.barh(y_pos, top_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14)
    for i,v in enumerate(top_vals):
        ax.text(v, i, f"{v:.3f}", va='center', ha='left', fontsize=9)

plt.tight_layout()
plt.savefig("SHAP_4Variants_decoded_2x2.png", dpi=600)
plt.show()

print("\n✅ 2×2 decoded SHAP feature‑importance plot saved as SHAP_4Variants_decoded_2x2.png")
