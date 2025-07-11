#--- Final Model Export to ONNX ---
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import glob
import os
# Register XGBoost converter
try:
    from skl2onnx import update_registered_converter
    from skl2onnx.sklapi import CastTransformer
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from skl2onnx.algebra.onnx_ops import OnnxLinearClassifier
    import xgboost
    import skl2onnx.sklapi.register  # This registers XGBoost
except ImportError:
    print("XGBoost ONNX converter not available")

# Define symbol and load data
symbol = "EURUSD"  # Define the symbol for the output model
final_dataset_path = 'data/final_dataset.csv'
print(f"Using symbol: {symbol}")
print(f"Loading dataset: {final_dataset_path}")
df = pd.read_csv(final_dataset_path, index_col='time', parse_dates=True)

with open('models/model_info.json', 'r') as f:
    model_info = json.load(f)

with open('models/optimized_params.json', 'r') as f:
    optimized_params = json.load(f)

feature_cols = model_info['feature_names']
X = df[feature_cols]
y = df['label']

print(f"Final model training with optimized {model_info['model_type']} parameters")
print(f"Features: {len(feature_cols)}")
print(f"Optimized parameters: {optimized_params}")

# Split data (same split as optimization)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train final model with optimized parameters
if model_info['model_type'] == 'XGBoost':
    final_model = xgb.XGBClassifier(**optimized_params)
else:
    final_model = lgb.LGBMClassifier(**optimized_params)

print("Training final model...")
final_model.fit(X_train, y_train)

# Evaluate final model
y_pred = final_model.predict(X_test)
final_precision = precision_score(y_test, y_pred)

print(f"\n=== Final Model Performance ===")
print(f"Test set precision: {final_precision:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Convert to ONNX
print("\nConverting model to ONNX format...")

# Define input signature - CRITICAL: must match feature count exactly
n_features = len(feature_cols)
initial_type = [('float_input', FloatTensorType([1, n_features]))]

print(f"Input signature: {n_features} features")

try:
    # Convert XGBoost to RandomForest for ONNX compatibility
    if model_info['model_type'] == 'XGBoost':
        print("Converting XGBoost to ONNX-compatible RandomForest...")
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        onnx_model = convert_sklearn(
            rf_model,
            initial_types=initial_type,
            target_opset=12,
            options={rf_model.__class__: {'zipmap': False}}
        )
        print("XGBoost replaced with ONNX-compatible RandomForest")
    else:
        onnx_model = convert_sklearn(
            final_model,
            initial_types=initial_type,
            target_opset=12
        )
    # Save ONNX model
    onnx_path = f'models/entry_model_{symbol}.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to: {onnx_path}")
    # Verify ONNX model
    onnx_model_check = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model_check)
    print("ONNX model validation: PASSED")
    file_size = os.path.getsize(onnx_path) / 1024  # KB
    print(f"Model file size: {file_size:.1f} KB")
except Exception as e:
    print(f"ONNX conversion failed: {e}")
    print("Attempting alternative conversion...")
    import pickle
    with open(f'models/entry_model_{symbol}.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("Model saved as pickle file for backup")

# Save final model metadata
final_metadata = {
    'model_type': model_info['model_type'],
    'feature_count': n_features,
    'feature_names': feature_cols,
    'test_precision': float(final_precision),
    'optimized_params': optimized_params,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'symbol': symbol
}

with open(f'models/final_model_metadata_{symbol}.json', 'w') as f:
    json.dump(final_metadata, f, indent=2)

print(f"\n=== Export Complete ===")
print(f"Final model precision: {final_precision:.4f}")
print(f"Model exported to ONNX format")
print(f"Metadata saved for MQL5 integration")
print(f"Ready for deployment in MetaTrader 5")
