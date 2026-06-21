# run_drift_check.py
# Run: python run_drift_check.py

import joblib
import pandas as pd
import numpy as np

from src.monitoring.drift_detector import DriftDetector

print("=" * 50)
print("DRIFT DETECTION CHECK")
print("=" * 50)

# Load training data as reference
(
    X_train, X_val, X_test,
    y_train, y_val, y_test
) = joblib.load('data/processed/data_splits.pkl')

# Initialize detector with training data
detector = DriftDetector(reference_data=X_train)

# Simulate drift: use test data as "current production data"
# In real world this would be today's incoming customers
print("\n📊 Checking test set for drift vs training data...")
report = detector.generate_drift_report(
    current_data=X_test,
    save_path="models/drift_report.json"
)

# Simulate HIGH drift scenario for demo
print("\n" + "─" * 45)
print("SIMULATING PRODUCTION DRIFT SCENARIO")
print("─" * 45)

# Create artificially drifted data
X_drifted = X_test.copy()

# Simulate: monthly charges increased by 40% company-wide
X_drifted['MonthlyCharges'] = (
    X_drifted['MonthlyCharges'] * 1.4
)
X_drifted['annual_revenue'] = (
    X_drifted['annual_revenue'] * 1.4
)
# Simulate: customers are newer (tenure dropped)
X_drifted['tenure'] = X_drifted['tenure'] * 0.6

print(
    "\n⚠️  Simulated scenario: "
    "Prices increased 40%, customer tenure dropped 40%"
)

drifted_report = detector.generate_drift_report(
    current_data=X_drifted,
    save_path="models/drift_report_simulated.json"
)

print("\n" + "=" * 50)
print("✅ DRIFT DETECTION COMPLETE!")
print(
    "\n💡 In production, this script would run daily"
    "\n   and alert the team when drift is detected."
)
print("=" * 50)