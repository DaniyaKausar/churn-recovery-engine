# src/monitoring/drift_detector.py
# ==================================
# Detects when real-world data changes from training data.
# Critical for production ML - models degrade silently
# when data distribution shifts.
#
# Example: If average monthly charges increase by 30%
# after a price hike, the model may become unreliable.

import pandas as pd
import numpy as np
from scipy import stats
import joblib
import json
import os

def make_json_safe(obj):
    """
    Converts numpy types to native Python types
    so they can be saved as JSON.
    """
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    return obj


class DriftDetector:
    """
    Statistical drift detection using KS test.
    
    KS Test (Kolmogorov-Smirnov):
    Compares two distributions and tells us if they
    are significantly different.
    
    p-value < 0.05 = distributions are different = DRIFT!
    p-value >= 0.05 = distributions are similar = OK
    
    We use this instead of Evidently to avoid
    NumPy compatibility issues.
    """

    def __init__(self, reference_data: pd.DataFrame):
        """
        reference_data: your training data
        This is our baseline - what 'normal' looks like
        """
        self.reference_data = reference_data
        self.numerical_cols = reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        print(
            f"✅ DriftDetector initialized with "
            f"{len(self.reference_data)} reference samples"
        )

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        significance_level: float = 0.05
    ) -> dict:
        """
        Compares current data against reference data.

        Returns:
        - drift_detected: True/False overall
        - drift_score: % of features that drifted
        - drifted_features: which features changed
        - feature_details: p-value for each feature
        """

        drifted_features = []
        feature_details = {}
        total_tested = 0

        for col in self.numerical_cols:
            if col not in current_data.columns:
                continue

            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()

            if len(curr_values) < 10:
                continue

            # KS test: are these from same distribution?
            ks_stat, p_value = stats.ks_2samp(
                ref_values,
                curr_values
            )

            is_drifted = bool(p_value < significance_level)
            total_tested += 1

            feature_details[col] = {
                'ks_statistic': round(float(ks_stat), 4),
                'p_value': round(float(p_value), 4),
                'drift_detected': is_drifted,
                'reference_mean': round(
                    float(ref_values.mean()), 2
                ),
                'current_mean': round(
                    float(curr_values.mean()), 2
                ),
                'mean_change_pct': round(
                    (curr_values.mean() - ref_values.mean())
                    / (abs(ref_values.mean()) + 0.1) * 100,
                    1
                ),
                'mean_shift_absolute': round(
                    float(curr_values.mean() - ref_values.mean()), 4
                )
            }

            if is_drifted:
                drifted_features.append(col)

        drift_score = (
            len(drifted_features) / total_tested
            if total_tested > 0 else 0
        )

        result = {
            'drift_detected': bool(len(drifted_features) > 0),
            'drift_score': round(drift_score, 3),
            'drifted_features_count': len(drifted_features),
            'total_features_tested': total_tested,
            'drifted_features': drifted_features,
            'feature_details': feature_details,
            'alert_level': self._get_alert_level(
                drift_score
            )
        }

        return result

    def _get_alert_level(self, drift_score: float) -> str:
        if drift_score >= 0.3:
            return "CRITICAL - Consider retraining"
        elif drift_score >= 0.15:
            return "WARNING - Monitor closely"
        elif drift_score > 0:
            return "LOW - Minor drift detected"
        else:
            return "OK - No drift detected"

    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        save_path: str = "models/drift_report.json"
    ) -> dict:
        """
        Generates and saves a full drift report.
        In production this would run daily/weekly.
        """
        print("\n🔍 Running drift detection...")
        result = self.detect_drift(current_data)

        # Save report
        # Save report (converted to JSON-safe types)
        os.makedirs('models', exist_ok=True)
        safe_result = make_json_safe(result)
        with open(save_path, 'w') as f:
            json.dump(safe_result, f, indent=2)

        # Print summary
        print(f"\n📊 Drift Detection Report:")
        print(f"   Alert Level: {result['alert_level']}")
        print(
            f"   Drift Score: "
            f"{result['drift_score']*100:.1f}% features drifted"
        )
        print(
            f"   Features drifted: "
            f"{result['drifted_features_count']}/"
            f"{result['total_features_tested']}"
        )

        if result['drifted_features']:
            print(f"\n   ⚠️  Drifted features:")
            for feat in result['drifted_features']:
                details = result['feature_details'][feat]
                print(
                    f"   - {feat}: "
                    f"shifted by {details['mean_shift_absolute']:+.3f} "
                    f"(scaled units), p-value={details['p_value']:.4f}"
                )

        print(f"\n💾 Report saved to {save_path}")
        return result