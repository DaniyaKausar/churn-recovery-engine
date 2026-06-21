# run_step5.py

import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.explainability.shap_explainer import SHAPExplainer

print("=" * 50)
print("STEP 5: SHAP Explainability")
print("=" * 50)

print("\n📂 Loading trained model and data...")

# Load booster directly (most compatible)
booster = xgb.Booster()
booster.load_model('models/churn_model.json')

feature_names = joblib.load('models/feature_names.pkl')
threshold = joblib.load('models/threshold.pkl')

(
    X_train, X_val, X_test,
    y_train, y_val, y_test
) = joblib.load('data/processed/data_splits.pkl')

print(f"✅ Model loaded")
print(f"✅ Features: {len(feature_names)} features")
print(f"✅ Test data: {len(X_test)} customers")
print(f"✅ Threshold: {threshold:.3f}")

# Create SHAP explainer using booster directly
explainer = SHAPExplainer(booster, feature_names)

# =============================================
# PART A: Global Analysis
# =============================================
print("\n" + "─" * 45)
print("PART A: Global SHAP Analysis")
print("─" * 45)

shap_values = explainer.compute_shap_values(X_test)

explainer.create_global_summary_plot(
    X_test,
    save_path="models/shap_summary.png"
)

explainer.create_feature_importance_plot(
    X_test,
    save_path="models/shap_importance.png"
)

insights = explainer.get_global_insights(X_test)

# =============================================
# PART B: Individual Customer Explanation
# =============================================
print("\n" + "─" * 45)
print("PART B: Individual Customer Explanation")
print("─" * 45)

# Get probabilities using DMatrix directly
dtest = xgb.DMatrix(X_test.values)
model_probs = booster.predict(dtest)

# Highest risk customer
high_risk_idx = model_probs.argmax()
high_risk_customer = X_test.iloc[[high_risk_idx]]
high_risk_prob = model_probs[high_risk_idx]

print(f"\n🎯 Analyzing highest risk customer:")
print(f"   Customer index: {high_risk_idx}")
print(f"   Churn probability: {high_risk_prob:.1%}")
print(f"   Actual outcome: "
      f"{'Churned ✓' if y_test.iloc[high_risk_idx] == 1 else 'Stayed'}")

explanation = explainer.explain_single_customer(
    high_risk_customer,
    customer_id=f"Customer_{high_risk_idx}"
)

print(f"\n📋 Explanation:")
print(explanation['explanation_text'])

explainer.create_waterfall_plot(
    high_risk_customer,
    customer_id="High Risk Customer",
    save_path="models/shap_waterfall.png"
)

# Lowest risk customer
low_risk_idx = model_probs.argmin()
low_risk_customer = X_test.iloc[[low_risk_idx]]
low_risk_prob = model_probs[low_risk_idx]

print(f"\n🟢 Lowest risk customer:")
print(f"   Churn probability: {low_risk_prob:.1%}")

low_explanation = explainer.explain_single_customer(
    low_risk_customer,
    customer_id=f"Customer_{low_risk_idx}"
)
print(f"\n📋 Explanation:")
print(low_explanation['explanation_text'])

# Save explainer
joblib.dump(explainer, 'models/shap_explainer.pkl')
print(f"\n💾 SHAP explainer saved")

print("\n" + "=" * 50)
print("✅ STEP 5 COMPLETE!")
print("   Files created:")
print("   📊 models/shap_summary.png")
print("   📊 models/shap_importance.png")
print("   📊 models/shap_waterfall.png")
print("=" * 50)
print("\n🎉 Ready for Step 6 — LLM Retention Strategy!")