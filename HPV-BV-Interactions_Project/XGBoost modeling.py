# Simplify the XGBoost classifier to speed up the training
xgb_classifier_simple = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Attempt to train and evaluate the simplified model
xgb_classifier_simple.fit(X_train_bv, y_train_bv_encoded_correct)

# Predict on the test set
y_pred_xgb_simple = xgb_classifier_simple.predict(X_test_bv)

# Evaluate the simplified XGBoost classifier
accuracy_xgb_simple = accuracy_score(y_test_bv_encoded_correct, y_pred_xgb_simple)
classification_report_xgb_simple = classification_report(y_test_bv_encoded_correct, y_pred_xgb_simple, target_names=encoder.classes_)

# Plotting feature importances for the simplified model
feature_importances_xgb_simple = xgb_classifier_simple.feature_importances_
sorted_idx_xgb_simple = feature_importances_xgb_simple.argsort()

plt.figure(figsize=(10, 7))
sns.barplot(x=feature_importances_xgb_simple[sorted_idx_xgb_simple], y=X_train_bv.columns[sorted_idx_xgb_simple])
plt.title('Feature Importance (Simplified XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

accuracy_xgb_simple, classification_report_xgb_simple
