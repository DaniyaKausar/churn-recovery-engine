# src/models/survival_model.py

from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

class ChurnSurvivalAnalyzer:
    """
    Predicts the TIME until churn using survival analysis.
    This answers: 'This customer has 73% chance of churning
    within the next 3 months'
    """
    
    def __init__(self):
        self.cox_model = CoxPHFitter()
        self.km_fitter = KaplanMeierFitter()
        
    def prepare_survival_data(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        survival_df = pd.DataFrame({
            'duration': df['tenure'],
            'event_observed': df['Churn'],
            'monthly_charges': df['MonthlyCharges'],
            'is_month_to_month': (
                df['Contract'] == 'Month-to-month'
            ).astype(int),
            'auto_payment': df['PaymentMethod'].isin([
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ]).astype(int),
            'service_count': df.get(
                'service_count', 
                pd.Series([0] * len(df))
            )
        })
        return survival_df
    
    def fit(self, survival_df: pd.DataFrame):
        # Overall survival curve
        self.km_fitter.fit(
            survival_df['duration'],
            event_observed=survival_df['event_observed']
        )
        
        # Cox model for individual predictions
        self.cox_model.fit(
            survival_df,
            duration_col='duration',
            event_col='event_observed',
            show_progress=False
        )
        
        print("✅ Survival model fitted")
        joblib.dump(self.cox_model, 'models/survival_model.pkl')
    
    def predict_churn_timeline(
        self, 
        customer_data: pd.DataFrame,
        time_horizons: list = [1, 3, 6, 12]
    ) -> dict:
        """
        Returns probability of churn at each time horizon
        e.g., {1: 0.12, 3: 0.34, 6: 0.56, 12: 0.78}
        """
        survival_func = self.cox_model.predict_survival_function(
            customer_data
        )
        
        churn_probs = {}
        for t in time_horizons:
            if t <= survival_func.index.max():
                # P(churn by time t) = 1 - S(t)
                survival_prob = float(
                    survival_func.iloc[
                        survival_func.index.get_indexer(
                            [t], method='nearest'
                        )
                    ]
                )
                churn_probs[f'{t}_month'] = round(
                    1 - survival_prob, 3
                )
        
        return churn_probs