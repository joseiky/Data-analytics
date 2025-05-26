import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load the dataset
file_path = '/mnt/e/Student\'s works/MDL articles/HPV-BV article/New version/HPV_BV_PAP_Final.xlsx'
sheet_name = 'Sheet2'

# Load the data
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Isolating features and target variables for BV Status and Cervical Cytology
features = data.drop(columns=['BV Status', 'NILM', 'RCC', 'ECA:AGC', 'ECA:ASCUS', 'ECA:ASC-H', 'ECA:HSIL', 'ECA:LSIL'])
target_bv = data['BV Status']
target_cytology = data[['NILM', 'RCC', 'ECA:AGC', 'ECA:ASCUS', 'ECA:ASC-H', 'ECA:HSIL', 'ECA:LSIL']].idxmax(axis=1)

# Preprocessing pipeline
numeric_features = features.select_dtypes(include=['float64', 'int64']).columns
categorical_features = ['Provider State', 'Pt. Gender']

# Preprocessor: Scaling numeric data and encoding categorical variables
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Create pipeline
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_model)])

# Split data into training and testing sets for BV Status and Cervical Cytology
X_train_bv, X_test_bv, y_train_bv, y_test_bv = train_test_split(features, target_bv, test_size=0.2, random_state=42)
X_train_cyto, X_test_cyto, y_train_cyto, y_test_cyto = train_test_split(features, target_cytology, test_size=0.2, random_state=42)

# --- BV Status Model ---
# Fit the model for BV Status
xgb_pipeline.fit(X_train_bv, y_train_bv)
y_pred_bv = xgb_pipeline.predict(X_test_bv)

# Save BV Status report and feature importance
report_bv = classification_report(y_test_bv, y_pred_bv, output_dict=True)
report_bv_df = pd.DataFrame(report_bv).transpose()

# Extract BV feature importance
feature_names = np.hstack([numeric_features, preprocessor.transformers_[1][1].get_feature_names_out()])
bv_importance = pd.DataFrame({'Feature': feature_names, 'XGBoost Importance': xgb_pipeline.named_steps['classifier'].feature_importances_})

# --- Cervical Cytology Model ---

# Label encoding for Cervical Cytology target
label_encoder_cyto = LabelEncoder()
y_train_cyto_encoded = label_encoder_cyto.fit_transform(y_train_cyto)
y_test_cyto_encoded = label_encoder_cyto.transform(y_test_cyto)

# Fit the model for Cervical Cytology with encoded labels
xgb_pipeline.fit(X_train_cyto, y_train_cyto_encoded)
y_pred_cyto = xgb_pipeline.predict(X_test_cyto)

# Save Cervical Cytology report with decoded labels
y_pred_cyto_decoded = label_encoder_cyto.inverse_transform(y_pred_cyto)
report_cyto = classification_report(y_test_cyto, y_pred_cyto_decoded, output_dict=True)
report_cyto_df = pd.DataFrame(report_cyto).transpose()

# Extract Cytology feature importance
cyto_importance = pd.DataFrame({'Feature': feature_names, 'XGBoost Importance': xgb_pipeline.named_steps['classifier'].feature_importances_})

# --- Save Results ---
# Save both BV and Cytology feature importance into separate sheets in the same file
output_file = '/mnt/e/Student\'s works/MDL articles/HPV-BV article/New version/xgboost_feature_importance_separated.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    report_bv_df.to_excel(writer, sheet_name='XGBoost_BV_Report', index=True)
    report_cyto_df.to_excel(writer, sheet_name='XGBoost_Cytology_Report', index=True)
    bv_importance.to_excel(writer, sheet_name='XGBoost_BV_Importance', index=False)
    cyto_importance.to_excel(writer, sheet_name='XGBoost_Cytology_Importance', index=False)

print("XGBoost model and feature importance saved successfully with separated sheets for BV and Cytology.")
