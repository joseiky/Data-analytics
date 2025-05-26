import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Cleaned HPV-BV data.xlsx', sheet_name=0)

# Define a function to prepare the data
def prepare_data(df, target, exclude):
    X = df.drop([exclude, target], axis=1)
    y = df[target]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Define a function to evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test, title):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{title} Accuracy: {accuracy * 100:.2f}%")
    
    # Handling for feature importances or coefficients
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    
    features_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    print(features_df.sort_values(by='Importance', ascending=False))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {title}")
    features_df.set_index('Feature')['Importance'].sort_values().plot(kind='barh')
    plt.show()

# BV Status Prediction
X_train_bv, X_test_bv, y_train_bv, y_test_bv = prepare_data(df, 'BV Status', 'Sample #')
models = [
    (RandomForestClassifier(random_state=42), "Random Forest - BV"),
    (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost - BV"),
    (LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression - BV")
]

for model, name in models:
    evaluate_model(model, X_train_bv, X_test_bv, y_train_bv, y_test_bv, name)

# Cervical Cancer Prediction
X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_data(df, "CERVICAL CYTOLOGY", 'Sample #')
models_cc = [
    (RandomForestClassifier(random_state=42), "Random Forest - Cervical Cancer"),
    (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost - Cervical Cancer"),
    (LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression - Cervical Cancer")
]

for model, name in models_cc:
    evaluate_model(model, X_train_cc, X_test_cc, y_train_cc, y_test_cc, name)
