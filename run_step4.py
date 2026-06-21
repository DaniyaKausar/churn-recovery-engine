# run_step4.py
# Run this file to train the XGBoost model
# Command: python run_step4.py

import joblib
import warnings
warnings.filterwarnings('ignore')

from src.models.churn_classifier import ChurnClassifier
from src.models.experiment_tracker import ExperimentTracker

print("=" * 50)
print("STEP 4: XGBoost Model Training")
print("=" * 50)

# Load the data splits we saved in Step 3
print("\n📂 Loading prepared data splits...")
(
    X_train, X_val, X_test, 
    y_train, y_val, y_test
) = joblib.load('data/processed/data_splits.pkl')

print(f"   ✅ Training data:   {X_train.shape}")
print(f"   ✅ Validation data: {X_val.shape}")
print(f"   ✅ Test data:       {X_test.shape}")

# Create and train model
classifier = ChurnClassifier()

# Step 4a: Find best settings + train model
classifier.tune_and_train(
    X_train, 
    y_train,
    n_trials=30
)

# Step 4b: Find optimal threshold
classifier.optimize_threshold(X_val, y_val)

# Step 4c: Evaluate on test set
metrics = classifier.evaluate(X_test, y_test)

# Step 4d: Show feature importance
importance_df = classifier.get_feature_importance()

# Step 4e: Save everything
classifier.save_model()

# Save metrics for later reference
joblib.dump(metrics, 'models/metrics.pkl')

print("\n" + "=" * 50)
print("✅ STEP 4 COMPLETE!")
print(f"   AUC-ROC: {metrics['auc_roc']}")
print(f"   F1 Score: {metrics['f1_score']}")
print(f"   Recall: {metrics['recall']}")
print("=" * 50)
print("\n📊 Logging to MLflow...")
tracker = ExperimentTracker()
run_id = tracker.log_training_run(
    params=classifier.best_params,
    metrics=metrics,
    model=classifier.booster,
    feature_names=classifier.feature_names,
    threshold=classifier.threshold
)
print(f"✅ Experiment logged! Run ID: {run_id}")
print(
    "\n💡 To view MLflow UI, open new terminal and run:"
    "\n   mlflow ui"
    "\n   Then open: http://localhost:5000"
)
print("\n🎉 Ready for Step 5 — SHAP Explainability!")