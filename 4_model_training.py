#--- Model Training & Validation Module ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Load the single labeled dataset
df = pd.read_csv('data/final_dataset.csv', index_col='time', parse_dates=True)
print(f"Loaded {len(df)} samples from the final dataset.")

# Separate features and target
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'ATR_14', 
                'support_100', 'resistance_100', 'dist_to_support_atr', 
                'dist_to_resistance_atr', 'is_london_session', 'is_ny_session', 'RSI_14_H1']

X = df[feature_cols]
y = df['label']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data: {len(X_train)} samples")
print(f"Validation data: {len(X_val)} samples")

print(f"Features: {len(feature_cols)}")
print(f"Training class distribution: {y_train.value_counts().to_dict()}")
print(f"Validation class distribution: {y_val.value_counts().to_dict()}")

# Train XGBoost baseline
print("\n=== Training XGBoost Baseline ===")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)  # Evaluate on validation set

print("XGBoost Results:")
print(classification_report(y_val, xgb_pred))

# Train LightGBM baseline
print("\n=== Training LightGBM Baseline ===")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_val)  # Evaluate on validation set

print("LightGBM Results:")
print(classification_report(y_val, lgb_pred))

# Compare models
from sklearn.metrics import precision_score, recall_score, f1_score

xgb_precision = precision_score(y_val, xgb_pred)
lgb_precision = precision_score(y_val, lgb_pred)

print(f"\n=== Model Comparison ===")
print(f"XGBoost Precision (Class 1): {xgb_precision:.4f}")
print(f"LightGBM Precision (Class 1): {lgb_precision:.4f}")

# Select best model
if xgb_precision >= lgb_precision:
    best_model = xgb_model
    best_name = "XGBoost"
    print(f"Selected: XGBoost (Precision: {xgb_precision:.4f})")
else:
    best_model = lgb_model
    best_name = "LightGBM"
    print(f"Selected: LightGBM (Precision: {lgb_precision:.4f})")

# Save model info for next phase
model_info = {
    'model_type': best_name,
    'precision': float(max(xgb_precision, lgb_precision)),
    'feature_count': len(feature_cols),
    'feature_names': feature_cols
}

import json
with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\nBaseline training complete. Best model: {best_name}")
print(f"Model info saved for optimization phase")
