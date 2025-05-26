# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y = y_encoded
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Load the dataset
df = pd.read_excel('/mnt/e/Student\'s works/MDL articles/HPV-BV article/Cleaned HPV-BV data.xlsx', sheet_name=0)

# Splitting the dataset into training and testing sets for BV Status prediction
X = df.drop(['Sample #', 'BV Status', "CERVICAL CYTOLOGY''"], axis=1)
y_bv = df['BV Status']

X_train_bv, X_test_bv, y_train_bv, y_test_bv = train_test_split(X, y_bv, test_size=0.2, random_state=42)

# Splitting the dataset into training and testing sets for Cervical Cytology prediction
y_cc = df["CERVICAL CYTOLOGY''"]

X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(X, y_cc, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\n{model_name} Classification Report:')
    print(classification_report(y_test, y_pred))
    print(f'{model_name} Accuracy:', accuracy_score(y_test, y_pred))
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:  # For models without feature_importances_ attribute
        perm_importance = permutation_importance(model, X_test, y_test)
        importances = perm_importance.importances_mean
    
    indices = np.argsort(importances)[::-1]
    print(f"{model_name} Feature Importances:")
    for i in range(min(10, len(indices))):  # Show top 10 features
        print(f"{i + 1}. feature {X_train.columns[indices[i]]} ({importances[indices[i]]:.4f})")
    
    # Visualization of feature importances
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} Feature Importances")
    plt.bar(range(min(10, X_train.shape[1])), importances[indices][:10], align='center')
    plt.xticks(range(min(10, X_train.shape[1])), X_train.columns[indices][:10], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Training and Evaluation for BV Status
for name, model in models.items():
    train_evaluate(model, X_train_bv, X_test_bv, y_train_bv, y_test_bv, name)

# Training and Evaluation for Cervical Cytology
for name, model in models.items():
    train_evaluate(model, X_train_cc, X_test_cc, y_train_cc, y_test_cc, name)
