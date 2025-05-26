import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# === CONFIG ===
input_file = "Empty_variant_rows_removed.xlsx"
original_file = "Sensitivity_Original_Data.xlsx"
filtered_file = "Sensitivity_Filtered_Data.xlsx"
output_dir = "sensitivity_outputs"
os.makedirs(output_dir, exist_ok=True)

# === LOAD AND PREP ===
df = pd.read_excel(input_file)

# Save original
df.to_excel(original_file, index=False)

# Filtered: exclude 'NP' ethnicity + missing Variant/Zygosity
df_filtered = df[
    (df["Ethnicity"].notna()) &
    (df["Ethnicity"] != "NP") &
    (df["Variant"].notna()) &
    (df["Zygosity"].notna())
]
df_filtered.to_excel(filtered_file, index=False)

# === TARGET & FEATURES ===
target = "Abnormal Flag"
features = [
    "AGE", "Pt Gender", "Ethnicity", "Specimen", "Source",
    "Gene Transcript", "Variant", "mRNA", "Protein", "Common Name",
    "Inheritance", "Location", "Test Name"
]

# === MODEL RUNNER ===
def run_models(df, tag):
    df[target] = df[target].fillna("N")
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = LabelEncoder().fit_transform(df[target])

    cat = X.select_dtypes("object").columns.tolist()
    num = X.select_dtypes("number").columns.tolist()

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num)
    ])

    pipe_lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe_xgb = Pipeline([("pre", pre), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # === Logistic Regression ===
    pipe_lr.fit(X_train, y_train)
    coefs = pipe_lr.named_steps["clf"].coef_[0]
    features_out = pipe_lr.named_steps["pre"].get_feature_names_out()
    pd.DataFrame({
        "Feature": features_out,
        "OddsRatio": np.exp(coefs),
        "Coefficient": coefs
    }).sort_values("OddsRatio", ascending=False).head(20).to_csv(
        f"{output_dir}/LR_{tag}_forestplot.csv", index=False
    )

    # === XGB ===
    pipe_xgb.fit(X_train, y_train)
    y_proba = pipe_xgb.predict_proba(X_test)[:, 1]
    y_pred = pipe_xgb.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ROC_{tag}.png", dpi=300)
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion – {tag}\nAccuracy = {acc:.2%}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Confusion_{tag}.png", dpi=300)
    plt.close()

    if tag == "filtered":
        X_test_trans = pipe_xgb.named_steps["pre"].transform(X_test)
        explainer = shap.Explainer(pipe_xgb.named_steps["clf"])
        shap_vals = explainer(X_test_trans)
        shap.summary_plot(
            shap_vals.values, features=X_test_trans,
            feature_names=features_out, plot_type="bar", show=False
        )
        plt.title(f"SHAP Summary – {tag}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/SHAP_{tag}.png", dpi=300)
        plt.close()

# === RUN BOTH ===
run_models(df, "original")
run_models(df_filtered, "filtered")

print("✅ Sensitivity analysis complete. All files saved to:", output_dir)

