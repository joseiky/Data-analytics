import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Define the base directory path
base_directory = '/mnt/e/Student\'s works/MDL articles/HPV-BV article/New version/'

# Create a subfolder for saving images
image_directory = os.path.join(base_directory, 'feature_importance_images')
os.makedirs(image_directory, exist_ok=True)

# Load the dataset from Sheet2
file_path = os.path.join(base_directory, 'HPV_BV_PAP_Final.xlsx')
df = pd.read_excel(file_path, sheet_name='Sheet2')

# Remove columns related to Ct scores
ct_columns = [col for col in df.columns if 'Ct' in col]
df = df.drop(columns=ct_columns)

# Encode categorical columns
le = LabelEncoder()
df['Provider State'] = le.fit_transform(df['Provider State'].astype(str))
df['Pt. Gender'] = le.fit_transform(df['Pt. Gender'].astype(str))

# Remove any remaining missing values
df = df.dropna()

# Define features (X) and target (y) variables
X = df.drop(columns=['RCC', 'ECA:AGC', 'ECA:ASCUS', 'ECA:ASC-H', 'ECA:HSIL', 'ECA:LSIL'])
y = df[['ECA:AGC', 'ECA:ASCUS', 'ECA:ASC-H', 'ECA:HSIL', 'ECA:LSIL']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features to help with model convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model with increased iterations
lr_model = MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=2000))
lr_model.fit(X_train_scaled, y_train)
lr_importances = np.mean([model.coef_[0] for model in lr_model.estimators_], axis=0)

# Train the XGBoost model with optimized parameters
xgb_model_optimized = MultiOutputClassifier(XGBClassifier(
    random_state=42, 
    max_depth=3,  
    n_estimators=50,  
    learning_rate=0.1, 
    use_label_encoder=False, 
    eval_metric='logloss'
))
xgb_model_optimized.fit(X_train_scaled, y_train)
xgb_importances_optimized = xgb_model_optimized.estimators_[0].feature_importances_

# Function to plot feature importances
def plot_feature_importances(importances, feature_names, model_name):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.title(f"Feature Importances ({model_name})")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, f'{model_name}_feature_importances.png'))
    plt.show()

# Plot and save the feature importance for both models
plot_feature_importances(lr_importances, X.columns, "Logistic Regression (Final Features)")
plot_feature_importances(xgb_importances_optimized, X.columns, "XGBoost (Optimized Features)")

# Save the feature importances to CSV files
pd.DataFrame({'Feature': X.columns, 'Logistic Regression': lr_importances}).to_csv(os.path.join(base_directory, 'feature_importances_lr_final.csv'), index=False)
pd.DataFrame({'Feature': X.columns, 'XGBoost': xgb_importances_optimized}).to_csv(os.path.join(base_directory, 'feature_importances_xgb_optimized_final.csv'), index=False)

