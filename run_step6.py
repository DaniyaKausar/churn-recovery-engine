# run_step6.py
# Run this to test the LLM Retention Strategy Engine
# Command: python run_step6.py

import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.retention.strategy_engine import RetentionStrategyEngine

print("=" * 55)
print("STEP 6: LLM Retention Strategy Engine")
print("=" * 55)

# Load everything we need
print("\n📂 Loading models and data...")

booster = xgb.Booster()
booster.load_model('models/churn_model.json')

feature_names = joblib.load('models/feature_names.pkl')
threshold = joblib.load('models/threshold.pkl')
shap_explainer = joblib.load('models/shap_explainer.pkl')

(
    X_train, X_val, X_test,
    y_train, y_val, y_test
) = joblib.load('data/processed/data_splits.pkl')

# Also load original data to get customer profiles
import sys
sys.path.append('.')
from src.data.ingestion import DataIngestion
from src.data.preprocessing import FeatureEngineer
import joblib

df_original = pd.read_csv('data/raw/telco_churn.csv')
df_original['TotalCharges'] = pd.to_numeric(
    df_original['TotalCharges'], errors='coerce'
)
df_original['TotalCharges'] = df_original[
    'TotalCharges'
].fillna(df_original['MonthlyCharges'])

print("✅ All models loaded")
print(f"✅ Test customers available: {len(X_test)}")

# Get predictions for test set
dtest = xgb.DMatrix(X_test.values)
all_probs = booster.predict(dtest)

# Initialize retention engine
print("\n🤖 Initializing Retention Strategy Engine...")
retention_engine = RetentionStrategyEngine()

# =============================================
# TEST 1: High Risk Customer
# =============================================
print("\n" + "─" * 50)
print("TEST 1: Generating strategy for HIGH RISK customer")
print("─" * 50)

# Find highest risk customer
high_risk_idx = all_probs.argmax()
high_risk_prob = all_probs[high_risk_idx]
high_risk_features = X_test.iloc[[high_risk_idx]]

# Get SHAP explanation
explanation = shap_explainer.explain_single_customer(
    high_risk_features,
    customer_id=f"Customer_{high_risk_idx}"
)

# Get original customer data for profile
# Match by position in test set
original_idx = X_test.index[high_risk_idx]
customer_row = df_original.iloc[original_idx]

customer_profile = {
    'Contract': customer_row.get('Contract', 'Unknown'),
    'tenure': int(customer_row.get('tenure', 0)),
    'InternetService': customer_row.get(
        'InternetService', 'Unknown'
    ),
    'TechSupport': customer_row.get('TechSupport', 'Unknown'),
    'OnlineSecurity': customer_row.get(
        'OnlineSecurity', 'Unknown'
    ),
    'PaymentMethod': customer_row.get(
        'PaymentMethod', 'Unknown'
    ),
    'SeniorCitizen': int(
        customer_row.get('SeniorCitizen', 0)
    ),
}

monthly_charges = float(
    customer_row.get('MonthlyCharges', 65.0)
)

print(f"\n👤 Customer Profile:")
print(f"   Churn Probability: {high_risk_prob:.1%}")
print(f"   Contract: {customer_profile['Contract']}")
print(f"   Tenure: {customer_profile['tenure']} months")
print(f"   Monthly Charges: ${monthly_charges:.2f}")
print(f"   Annual Revenue at Risk: "
      f"${monthly_charges * 12:.0f}")

print(f"\n⚠️  Top Churn Risk Factors:")
for factor in explanation['top_risk_factors'][:3]:
    print(f"   - {factor['feature']}: "
          f"+{factor['shap_value']*100:.1f}% churn push")

# Generate strategy
print(f"\n🎯 Generating AI Retention Strategy...")
strategy = retention_engine.generate_strategy(
    customer_profile=customer_profile,
    churn_probability=high_risk_prob,
    top_risk_factors=explanation['top_risk_factors'],
    top_protective_factors=explanation[
        'top_protective_factors'
    ],
    monthly_charges=monthly_charges
)

print(f"\n{'='*50}")
print(f"🎯 RETENTION STRATEGY GENERATED:")
print(f"{'='*50}")
print(f"  Urgency Level:    {strategy['urgency_level']}")
print(f"  Contact Channel:  {strategy.get('channel', 'N/A')}")
print(f"  Act Within:       "
      f"{strategy['action_deadline_days']} days")
print(f"  Best Time:        "
      f"{strategy.get('best_time_to_contact', 'N/A')}")
print(f"\n  Primary Action:")
print(f"  → {strategy['primary_action']}")
print(f"\n  Offer to Make:")
print(f"  → {strategy['offer']}")
print(f"\n  Talking Points:")
for i, point in enumerate(
    strategy.get('talking_points', []), 1
):
    print(f"  {i}. {point}")
print(f"\n  Expected Outcome:")
print(f"  → Retention Probability: "
      f"{strategy['estimated_retention_probability']:.1%}")
print(f"  → Revenue Saved if Retained: "
      f"${strategy['estimated_revenue_saved']:,.0f}")
print(f"  → Revenue at Risk: "
      f"${strategy['annual_revenue_at_risk']:,.0f}/year")
print(f"\n  Strategy Reasoning:")
print(f"  → {strategy.get('reasoning', 'N/A')}")

# =============================================
# TEST 2: Medium Risk Customer
# =============================================
print(f"\n{'─'*50}")
print("TEST 2: Generating strategy for MEDIUM RISK customer")
print("─" * 50)

# Find a medium risk customer (probability between 0.4-0.7)
medium_mask = (all_probs >= 0.4) & (all_probs <= 0.7)
if medium_mask.any():
    medium_indices = np.where(medium_mask)[0]
    medium_idx = medium_indices[len(medium_indices)//2]
    medium_prob = all_probs[medium_idx]
    medium_features = X_test.iloc[[medium_idx]]
    
    medium_explanation = shap_explainer.explain_single_customer(
        medium_features,
        customer_id=f"Customer_{medium_idx}"
    )
    
    original_idx_med = X_test.index[medium_idx]
    customer_row_med = df_original.iloc[original_idx_med]
    
    medium_profile = {
        'Contract': customer_row_med.get(
            'Contract', 'Unknown'
        ),
        'tenure': int(customer_row_med.get('tenure', 0)),
        'InternetService': customer_row_med.get(
            'InternetService', 'Unknown'
        ),
        'TechSupport': customer_row_med.get(
            'TechSupport', 'Unknown'
        ),
        'OnlineSecurity': customer_row_med.get(
            'OnlineSecurity', 'Unknown'
        ),
        'PaymentMethod': customer_row_med.get(
            'PaymentMethod', 'Unknown'
        ),
        'SeniorCitizen': int(
            customer_row_med.get('SeniorCitizen', 0)
        ),
    }
    
    monthly_charges_med = float(
        customer_row_med.get('MonthlyCharges', 65.0)
    )
    
    print(f"\n👤 Customer Profile:")
    print(f"   Churn Probability: {medium_prob:.1%}")
    print(f"   Contract: {medium_profile['Contract']}")
    print(f"   Tenure: {medium_profile['tenure']} months")
    print(f"   Monthly Charges: ${monthly_charges_med:.2f}")
    
    print(f"\n🎯 Generating AI Retention Strategy...")
    strategy_med = retention_engine.generate_strategy(
        customer_profile=medium_profile,
        churn_probability=medium_prob,
        top_risk_factors=medium_explanation[
            'top_risk_factors'
        ],
        top_protective_factors=medium_explanation[
            'top_protective_factors'
        ],
        monthly_charges=monthly_charges_med
    )
    
    print(f"\n{'='*50}")
    print(f"🎯 RETENTION STRATEGY GENERATED:")
    print(f"{'='*50}")
    print(f"  Urgency Level:    {strategy_med['urgency_level']}")
    print(f"  Contact Channel:  "
          f"{strategy_med.get('channel', 'N/A')}")
    print(f"  Act Within:       "
          f"{strategy_med['action_deadline_days']} days")
    print(f"\n  Primary Action:")
    print(f"  → {strategy_med['primary_action']}")
    print(f"\n  Offer to Make:")
    print(f"  → {strategy_med['offer']}")
    print(f"\n  Talking Points:")
    for i, point in enumerate(
        strategy_med.get('talking_points', []), 1
    ):
        print(f"  {i}. {point}")
    print(f"\n  Expected Outcome:")
    print(f"  → Retention Probability: "
          f"{strategy_med['estimated_retention_probability']:.1%}")
    print(f"  → Revenue Saved: "
          f"${strategy_med['estimated_revenue_saved']:,.0f}")

# Save strategies for reference
# Save strategies for reference
import json

def convert_to_serializable(obj):
    """
    Convert numpy types to Python native types
    so JSON can serialize them.
    numpy float32 → Python float
    numpy int64 → Python int
    """
    import numpy as np
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {
            k: convert_to_serializable(v) 
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [
            convert_to_serializable(i) for i in obj
        ]
    return obj

# Convert before saving
strategy_clean = convert_to_serializable(strategy)
strategy_med_clean = convert_to_serializable(
    strategy_med if medium_mask.any() else {}
)

with open('models/sample_strategies.json', 'w') as f:
    json.dump({
        'high_risk_strategy': strategy_clean,
        'medium_risk_strategy': strategy_med_clean
    }, f, indent=2)

print(f"\n💾 Sample strategies saved to "
      f"models/sample_strategies.json")

print("\n" + "=" * 55)
print("✅ STEP 6 COMPLETE!")
print("=" * 55)
print("\n🎉 Ready for Step 7 — FastAPI Backend!")