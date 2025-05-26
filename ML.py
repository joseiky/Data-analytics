import pandas as pd
import numpy as np
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# === Load data ===
df = pd.read_excel("Empty_variant_rows_removed.xlsx")

# === Define targets ===
targets = {
    "Abnormal Flag": "Abnormal_Flag_Target",
    "Zygosity": "Zygosity_Target",
    "Result": "Result_Target"
}
df["Abnormal_Flag_Target"] = df["Abnormal Flag"].fillna("N")
df["Zygosity_Target"] = df["Zygosity"].fillna("Unknown")
df["Result_Target"] = df["Result"].fillna("No Abnormality")

# === Define features ===
features = [
    "AGE", "Pt Gender", "Ethnicity", "Specimen", "Source",
    "Gene Transcript", "Variant", "mRNA", "Protein", "Common Name",
    "Inheritance", "Location", "Date Specimen Collected", "Test Name"
]
df = df.dropna(subset=features + list(targets.values()))
df["Year"] = pd.to_datetime(df["Date Specimen Collected"]).dt.year
df["Month"] = pd.to_datetime(df["Date Specimen Collected"]).dt.month
features = [f for f in features if f != "Date Specimen Collected"]
features += ["Year", "Month"]

# === Preprocessing pipeline ===
X = df[features]
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# === Define models ===
models = {
    "LR": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=100),
    "DT": DecisionTreeClassifier(),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# === Output folder ===
os.makedirs("ml_outputs", exist_ok=True)

# === Loop through targets and models ===
for target_label, target_col in targets.items():
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    for model_name, model in models.items():
        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", model)
        ])

        # Check class balance for stratification
        class_counts = pd.Series(y_enc).value_counts()
        if (class_counts < 2).any():
            print(f"⚠️ Skipping {model_name} on {target_col} due to class imbalance.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.3, stratify=y_enc, random_state=42
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if len(np.unique(y_enc)) == 2 else None

        # Save predictions
        pred_df = pd.DataFrame({
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred)
        })
        if y_proba is not None:
            pred_df["y_proba"] = y_proba

        pred_df.to_csv(f"ml_outputs/{model_name}_{target_col}_predictions.csv", index=False)
        pd.DataFrame(confusion_matrix(y_test, y_pred)).to_csv(
            f"ml_outputs/{model_name}_{target_col}_confusion.csv", index=False
        )

        # Feature importance or coefficients
        feature_names = pipe.named_steps["pre"].get_feature_names_out()
        if model_name == "LR":
            coefs = pipe.named_steps["clf"].coef_[0]
            pd.DataFrame({
                "Feature": feature_names,
                "OddsRatio": np.exp(coefs),
                "Coefficient": coefs
            }).to_csv(f"ml_outputs/{model_name}_{target_col}_forestplot.csv", index=False)
        elif hasattr(pipe.named_steps["clf"], "feature_importances_"):
            importances = pipe.named_steps["clf"].feature_importances_
            pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).to_csv(f"ml_outputs/{model_name}_{target_col}_feature_importance.csv", index=False)

        # SHAP values (tree-based models only)
        if model_name in ["RF", "DT", "XGB"]:
            transformed_X = pipe.named_steps["pre"].transform(X_test)
            transformed_X = np.array(transformed_X, dtype=np.float32)

            explainer = shap.Explainer(pipe.named_steps["clf"])
            shap_values = explainer(transformed_X)

            if shap_values.values.ndim == 3:  # multiclass
                for i, cls in enumerate(le.classes_):
                    class_shap = pd.DataFrame(shap_values.values[:, :, i], columns=feature_names)
                    class_shap["y_true"] = y_test
                    class_shap.to_csv(f"ml_outputs/{model_name}_{target_col}_shap_class_{cls}.csv", index=False)
            else:  # binary
                shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
                shap_df["y_true"] = y_test
                shap_df.to_csv(f"ml_outputs/{model_name}_{target_col}_shap.csv", index=False)

print("✅ All models completed. Results saved in 'ml_outputs/' folder.")
