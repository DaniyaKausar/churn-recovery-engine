# src/explainability/shap_explainer.py
# =====================================
# SHAP = SHapley Additive exPlanations
#
# This file explains WHY the model predicted churn
# for each customer, using mathematically correct
# probability percentage points (not raw log-odds).

import shap
import pandas as pd
import numpy as np
import scipy.special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAPExplainer tells us WHY the model made each prediction,
    using proper probability percentage points.
    """

    def __init__(self, model, feature_names: list):
        print("🔧 Setting up SHAP explainer...")

        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        self.shap_values = None

        # Parse expected_value correctly regardless of format
        raw_expected = self.explainer.expected_value

        print(f"   Raw expected_value type: {type(raw_expected)}")
        print(f"   Raw expected_value: {raw_expected}")

        if isinstance(raw_expected, (list, np.ndarray)):
            self.base_value_logodds = float(raw_expected[0])
        else:
            self.base_value_logodds = float(raw_expected)

        base_prob = scipy.special.expit(
            self.base_value_logodds
        )
        print(f"   Base value (log-odds): "
              f"{self.base_value_logodds:.4f}")
        print(f"   Base value (probability): "
              f"{base_prob*100:.1f}%")

        print("✅ SHAP explainer ready!")

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes SHAP values for ALL customers in dataset.
        Used for global analysis.
        """
        print(f"\n🧮 Computing SHAP values for "
              f"{len(X)} customers...")
        print("   (This may take 1-2 minutes...)")

        self.shap_values = self.explainer.shap_values(X)

        print(f"✅ SHAP values computed!")
        print(f"   Shape: {self.shap_values.shape}")
        print(f"   Each row = one customer")
        print(f"   Each column = one feature's contribution")

        return self.shap_values

    def explain_single_customer(
        self,
        customer_features: pd.DataFrame,
        customer_id: str = "Customer"
    ) -> dict:
        """
        Explains prediction for ONE specific customer.
        Returns probability percentage point contributions
        that are always bounded and mathematically correct.
        """

        shap_vals = self.explainer.shap_values(
            customer_features
        )
        customer_shap = shap_vals[0]

        base_value_logodds = self.base_value_logodds
        base_probability = scipy.special.expit(
            base_value_logodds
        )

        final_logodds = base_value_logodds + customer_shap.sum()
        final_probability = scipy.special.expit(final_logodds)

        feature_shap_pairs = list(zip(
            self.feature_names,
            customer_shap,
            customer_features.values[0]
        ))

        feature_shap_pairs.sort(
            key=lambda x: abs(x[1]), reverse=True
        )

        feature_impacts = []
        running_logodds = base_value_logodds
        running_prob = base_probability

        for feature, shap_val, actual_val in feature_shap_pairs:
            new_logodds = running_logodds + shap_val
            new_prob = scipy.special.expit(new_logodds)

            prob_impact = new_prob - running_prob

            # For protective factors, store as positive "reduction" value
            # For risk factors, keep as positive "increase" value
            display_impact = abs(prob_impact * 100)

            feature_impacts.append({
                'feature': feature,
                'shap_value': float(shap_val),
                'probability_impact_pct': round(
                    float(display_impact), 2
                ),
                'actual_value': float(actual_val),
                'direction': (
                    'increases_churn_risk'
                    if shap_val > 0
                    else 'decreases_churn_risk'
                ),
                'abs_impact': abs(float(prob_impact))
            })

            running_logodds = new_logodds
            running_prob = new_prob

        feature_impacts.sort(
            key=lambda x: x['abs_impact'], reverse=True
        )

        risk_factors = [
            f for f in feature_impacts
            if f['direction'] == 'increases_churn_risk'
        ][:5]

        protective_factors = [
            f for f in feature_impacts
            if f['direction'] == 'decreases_churn_risk'
        ][:5]

        explanation_text = self._generate_explanation(
            risk_factors,
            protective_factors,
            base_probability,
            final_probability
        )

        return {
            'customer_id': customer_id,
            'base_churn_rate': round(base_probability, 4),
            'final_churn_probability': round(
                final_probability, 4
            ),
            'top_risk_factors': risk_factors,
            'top_protective_factors': protective_factors,
            'all_impacts': feature_impacts,
            'explanation_text': explanation_text
        }

    def _generate_explanation(
        self,
        risk_factors: list,
        protective_factors: list,
        base_rate: float,
        final_probability: float
    ) -> str:
        """
        Converts probability impacts into plain English.
        Shows both true dataset rate and model's internal baseline
        for full transparency.
        """
        
        lines = []
        lines.append(
            f"Actual dataset churn rate: 26.5% "
            f"(true average across all customers)"
        )
        lines.append(
            f"Model's internal baseline: {base_rate*100:.1f}% "
            f"(adjusted due to class-imbalance weighting "
            f"during training)"
        )
        lines.append(
            f"This customer's predicted churn probability: "
            f"{final_probability*100:.1f}%"
        )
        
        if risk_factors:
            lines.append("\n⚠️  Factors INCREASING churn risk:")
            for i, factor in enumerate(risk_factors[:3], 1):
                impact = factor['probability_impact_pct']
                lines.append(
                    f"  {i}. {factor['feature']}: "
                    f"+{impact:.1f} percentage points"
                )
        
        if protective_factors:
            lines.append(
                "\n✅ Factors PROTECTING against churn:"
            )
            for i, factor in enumerate(
                protective_factors[:3], 1
            ):
                impact = abs(factor['probability_impact_pct'])
                lines.append(
                    f"  {i}. {factor['feature']}: "
                    f"reduces risk by {impact:.1f} percentage points"
                )
        
        return "\n".join(lines)

    def create_global_summary_plot(
        self,
        X: pd.DataFrame,
        save_path: str = "models/shap_summary.png"
    ):
        print("\n📊 Creating SHAP summary plot...")

        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=(10, 8))

        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
            max_display=15,
            plot_type="dot"
        )

        plt.title(
            "SHAP Feature Impact on Churn Prediction\n"
            "(Red = High Feature Value, "
            "Blue = Low Feature Value)",
            fontsize=12,
            pad=20
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Summary plot saved to {save_path}")
        return save_path

    def create_feature_importance_plot(
        self,
        X: pd.DataFrame,
        save_path: str = "models/shap_importance.png"
    ):
        print("\n📊 Creating feature importance plot...")

        if self.shap_values is None:
            self.compute_shap_values(X)

        mean_shap = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_shap
        }).sort_values(
            'mean_abs_shap', ascending=True
        ).tail(15)

        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.barh(
            importance_df['feature'],
            importance_df['mean_abs_shap'],
            color='steelblue',
            alpha=0.8
        )

        for bar, val in zip(
            bars, importance_df['mean_abs_shap']
        ):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}',
                va='center',
                fontsize=9
            )

        ax.set_xlabel(
            'Mean |SHAP Value| (log-odds impact)',
            fontsize=11
        )
        ax.set_title(
            'Feature Importance via SHAP Values\n'
            'Churn Prediction Model',
            fontsize=13,
            fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Importance plot saved to {save_path}")
        return save_path

    def create_waterfall_plot(
        self,
        customer_features: pd.DataFrame,
        customer_id: str = "Sample Customer",
        save_path: str = "models/shap_waterfall.png"
    ):
        print(f"\n📊 Creating waterfall plot for "
              f"{customer_id}...")

        shap_explanation = self.explainer(customer_features)

        plt.figure(figsize=(12, 6))

        shap.plots.waterfall(
            shap_explanation[0],
            max_display=12,
            show=False
        )

        plt.title(
            f"Churn Prediction Explanation: {customer_id}\n"
            f"How each factor contributes to churn risk "
            f"(log-odds scale)",
            fontsize=11,
            pad=15
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Waterfall plot saved to {save_path}")
        return save_path

    def get_global_insights(self, X: pd.DataFrame) -> dict:
        if self.shap_values is None:
            self.compute_shap_values(X)

        mean_shap = np.abs(self.shap_values).mean(axis=0)

        feature_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)

        top_3 = feature_ranking.head(3)

        insights = {
            'top_churn_drivers': top_3.to_dict('records'),
            'feature_ranking': feature_ranking.to_dict(
                'records'
            ),
            'insight_summary': (
                f"Key finding: The top 3 churn predictors are "
                f"{top_3.iloc[0]['feature']}, "
                f"{top_3.iloc[1]['feature']}, and "
                f"{top_3.iloc[2]['feature']}. "
                f"Together they explain "
                f"{top_3['importance'].sum() / mean_shap.sum() * 100:.1f}% "
                f"of the model's decisions."
            )
        }

        print(f"\n💡 Key Insight:")
        print(f"   {insights['insight_summary']}")

        return insights