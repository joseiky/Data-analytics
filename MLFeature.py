import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Load cleaned dataset ===
df = pd.read_excel("Empty_variant_rows_removed.xlsx")

# === Targets ===
targets = {
    "Abnormal Flag": "Abnormal_Flag_Target",
    "Zygosity": "Zygosity_Target",
    "Result": "Result_Target"
}
df["Abnormal_Flag_Target"] = df["Abnormal Flag"].fillna("N")
df["Zygosity_Target"] = df["Zygosity"].fillna("Unknown")
df["Result_Target"] = df["Result"].fillna("No Abnormality")

# === Features ===
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

X = df[features]
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

models = {
    "RF": RandomForestClassifier(n_estimators=100),
    "DT": DecisionTreeClassifier(),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

os.makedirs("ml_outputs", exist_ok=True)

for target_name, target_col in targets.items():
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Skip targets with class imbalance
    class_counts = pd.Series(y_enc).value_counts()
    if (class_counts < 2).any():
        print(f"⚠️ Skipping {target_name} due to class imbalance.")
        continue

    X_train, _, y_train, _ = train_test_split(X, y_enc, test_size=0.3, stratify=y_enc, random_state=42)

    for model_name, model in models.items():
        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", model)
        ])
        pipe.fit(X_train, y_train)
        feature_names = pipe.named_steps["pre"].get_feature_names_out()
        importances = pipe.named_steps["clf"].feature_importances_

        pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).to_csv(f"ml_outputs/{model_name}_{target_col}_feature_importance.csv", index=False)

print("✅ Feature importance files saved in 'ml_outputs/' folder.")
