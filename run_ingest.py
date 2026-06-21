# run_step2.py
# Run this to test Steps 2 and 3 together
# Command: python run_step2.py

from src.data.ingestion import DataIngestion
from src.data.preprocessing import FeatureEngineer
import pandas as pd

print("=" * 50)
print("STEP 2: Data Ingestion")
print("=" * 50)

ingestion = DataIngestion(
    filepath='data/raw/telco_churn.csv'
)
df = ingestion.load()

print("\n" + "=" * 50)
print("STEP 3: Feature Engineering")
print("=" * 50)

# Create FeatureEngineer instance
fe = FeatureEngineer()

# Build new features
df_engineered = fe.build_features(df)

# Encode text columns + scale numbers
X, y = fe.encode_and_scale(df_engineered, fit=True)

# Split into train/val/test
X_train, X_val, X_test, y_train, y_val, y_test = (
    fe.split_data(X, y)
)

# Save splits for next steps
import joblib
joblib.dump(
    (X_train, X_val, X_test, y_train, y_val, y_test),
    'data/processed/data_splits.pkl'
)

print("\n💾 Saved data splits to "
      "data/processed/data_splits.pkl")

print("\n" + "=" * 50)
print("✅ STEP 3 COMPLETE!")
print(f"   Features created: {len(fe.feature_names)}")
print(f"   Feature names: {fe.feature_names}")
print("=" * 50)
print("\n🎉 Ready for Step 4 — XGBoost Model Training!")