# src/data/preprocessing.py
# =========================
# This file creates NEW features from existing columns
# and converts text columns to numbers for the ML model
#
# WHY do we need this?
# ML models only understand NUMBERS, not text like "Yes"/"No"
# Also, smart features = better predictions

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class FeatureEngineer:
    """
    FeatureEngineer does 3 things:
    1. Creates new meaningful features from existing columns
    2. Converts text (categorical) columns to numbers
    3. Scales numbers to same range so model treats them fairly
    """
    
    def __init__(self):
        # LabelEncoder converts text to numbers
        # e.g., 'Yes' -> 1, 'No' -> 0
        self.encoders = {}
        
        # StandardScaler makes all numbers same scale
        # e.g., tenure(0-72) and charges(18-118) become same range
        self.scaler = StandardScaler()
        
        # Will store list of feature column names
        self.feature_names = []
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features from existing columns.
        This is the most important function for model accuracy.
        """
        print("\n🔨 Building new features...")
        df = df.copy()  # Never modify original data
        
        # ================================================
        # FEATURE 1: Revenue at Risk
        # Business meaning: How much money do we lose if
        # this customer churns?
        # ================================================
        df['annual_revenue'] = df['MonthlyCharges'] * 12
        print("   ✅ Created: annual_revenue")
        
        # ================================================
        # FEATURE 2: Average Monthly Spend
        # Business meaning: What does customer pay on average
        # per month across their entire relationship?
        # We add 1 to tenure to avoid division by zero
        # ================================================
        df['avg_monthly_spend'] = (
            df['TotalCharges'] / (df['tenure'] + 1)
        )
        print("   ✅ Created: avg_monthly_spend")
        
        # ================================================
        # FEATURE 3: Charge Increase Rate
        # Business meaning: Is the customer paying MORE
        # now vs their historical average?
        # High value = recent price hike = churn risk
        # ================================================
        df['charge_increase_rate'] = (
            df['MonthlyCharges'] / 
            (df['avg_monthly_spend'] + 0.01)
        )
        print("   ✅ Created: charge_increase_rate")
        
        # ================================================
        # FEATURE 4: Service Count
        # Business meaning: How many services does the
        # customer use? More services = more engaged = 
        # less likely to churn
        # ================================================
        service_cols = [
            'PhoneService', 'MultipleLines', 
            'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies'
        ]
        
        # Count how many columns have 'Yes'
        df['service_count'] = df[service_cols].apply(
            lambda row: (row == 'Yes').sum(), 
            axis=1  # axis=1 means apply across columns
        )
        print("   ✅ Created: service_count")
        
        # ================================================
        # FEATURE 5: Engagement Score (0 to 1)
        # Business meaning: Normalized version of service_count
        # 0 = uses no services, 1 = uses all services
        # ================================================
        df['engagement_score'] = (
            df['service_count'] / len(service_cols)
        )
        print("   ✅ Created: engagement_score")
        
        # ================================================
        # FEATURE 6: Is Month-to-Month Contract?
        # Business meaning: Month-to-month customers can
        # leave anytime = highest churn risk
        # ================================================
        df['is_month_to_month'] = (
            df['Contract'] == 'Month-to-month'
        ).astype(int)
        print("   ✅ Created: is_month_to_month")
        
        # ================================================
        # FEATURE 7: Auto Payment
        # Business meaning: Customers with automatic payment
        # are less likely to churn (set it and forget it)
        # ================================================
        auto_payment_methods = [
            'Bank transfer (automatic)', 
            'Credit card (automatic)'
        ]
        df['auto_payment'] = df['PaymentMethod'].isin(
            auto_payment_methods
        ).astype(int)
        print("   ✅ Created: auto_payment")
        
        # ================================================
        # FEATURE 8: Tenure Segment
        # Business meaning: Group customers by loyalty stage
        # New customers churn differently than loyal ones
        # ================================================
        df['tenure_segment'] = pd.cut(
            df['tenure'],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=['new', 'early', 'growing', 
                   'loyal', 'champion']
        ).astype(str)
        print("   ✅ Created: tenure_segment")
        
        # ================================================
        # FEATURE 9: Has Internet Service?
        # Business meaning: Internet customers have more
        # options to switch providers
        # ================================================
        df['has_internet'] = (
            df['InternetService'] != 'No'
        ).astype(int)
        print("   ✅ Created: has_internet")
        
        # ================================================
        # FEATURE 10: Senior with No Support
        # Business meaning: Senior citizens without tech
        # support may be frustrated = churn risk
        # ================================================
        df['senior_no_support'] = (
            (df['SeniorCitizen'] == 1) & 
            (df['TechSupport'] == 'No')
        ).astype(int)
        print("   ✅ Created: senior_no_support")
        
        print(f"\n📊 Total features now: {len(df.columns)} columns")
        return df
    
    def encode_and_scale(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> tuple:
        print("\n🔢 Encoding categorical columns...")
        df = df.copy()

        # Find all text columns
        categorical_cols = df.select_dtypes(
            include=['object']
        ).columns.tolist()

        # Remove customerID
        categorical_cols = [
            col for col in categorical_cols
            if col != 'customerID'
        ]

        print(f"   Found {len(categorical_cols)} text columns")

        # Convert each text column to numbers
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(
                    df[col].astype(str)
                )
                self.encoders[col] = le
            else:
                if col in self.encoders:
                    le = self.encoders[col]
                    try:
                        df[col] = le.transform(
                            df[col].astype(str)
                        )
                    except ValueError:
                        # Handle unseen labels
                        df[col] = le.transform(
                            [le.classes_[0]] * len(df)
                        )

        print(f"   ✅ All text columns converted")

        # Define columns to drop
        cols_to_drop = ['customerID', 'Churn']

        if fit:
            # During training: learn the feature names
            self.feature_names = [
                col for col in df.columns
                if col not in cols_to_drop
            ]

        # ✅ KEY FIX: Always use saved feature_names order
        # This ensures prediction matches training EXACTLY
        X = df[self.feature_names]
        y = df['Churn'] if 'Churn' in df.columns else None

        print(f"\n⚖️  Scaling features...")

        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Convert back to DataFrame
        X_scaled = pd.DataFrame(
            X_scaled,
            columns=self.feature_names,
            index=X.index
        )

        print(f"   ✅ Scaling complete")

        if fit:
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.scaler, 'models/scaler.pkl')
            joblib.dump(self.encoders, 'models/encoders.pkl')
            joblib.dump(
                self.feature_names,
                'models/feature_names.pkl'
            )
            print(f"\n💾 Saved scaler, encoders, feature names")

        return X_scaled, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> tuple:
        """
        Splits data into 3 parts:
        - Training set (70%): Model learns from this
        - Validation set (10%): We tune model with this  
        - Test set (20%): Final evaluation (never seen by model)
        
        Think of it like:
        - Training = studying from textbook
        - Validation = practice tests
        - Test = final exam
        """
        print(f"\n✂️  Splitting data...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,      # Same split every time
            stratify=y            # Keep churn ratio same
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"   Training set:   {len(X_train)} customers "
              f"({len(X_train)/len(X)*100:.0f}%)")
        print(f"   Validation set: {len(X_val)} customers "
              f"({len(X_val)/len(X)*100:.0f}%)")
        print(f"   Test set:       {len(X_test)} customers "
              f"({len(X_test)/len(X)*100:.0f}%)")
        
        # Check churn ratio is preserved in each split
        print(f"\n   Churn rate in each split:")
        print(f"   Training:   {y_train.mean()*100:.1f}%")
        print(f"   Validation: {y_val.mean()*100:.1f}%")
        print(f"   Test:       {y_test.mean()*100:.1f}%")
        
        return X_train, X_val, X_test, y_train, y_val, y_test