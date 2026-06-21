# src/data/ingestion.py
# =====================
# This file is responsible for:
# 1. Loading the CSV dataset
# 2. Cleaning messy data types
# 3. Validating the data is correct before we use it

import pandas as pd
import numpy as np
from pathlib import Path


class DataIngestion:
    """
    DataIngestion class handles loading and basic cleaning
    of the Telco Customer Churn dataset.
    
    Think of this as the 'entry gate' of our ML pipeline.
    Nothing proceeds unless data passes through here cleanly.
    """
    
    def __init__(self, filepath: str):
        """
        filepath: path to your CSV file
        Example: 'data/raw/telco_churn.csv'
        """
        self.filepath = filepath
        self.df = None  # Will store our loaded dataframe
        
    def load(self) -> pd.DataFrame:
        """
        Main function that:
        1. Reads the CSV
        2. Cleans data types
        3. Validates the data
        4. Returns clean dataframe
        """
        print(f"📂 Loading data from: {self.filepath}")
        
        # Check if file exists before trying to load
        if not Path(self.filepath).exists():
            raise FileNotFoundError(
                f"❌ File not found: {self.filepath}\n"
                f"Make sure your CSV is in data/raw/ folder"
            )
        
        # Load the CSV file into a pandas DataFrame
        # A DataFrame is like an Excel spreadsheet in Python
        self.df = pd.read_csv(self.filepath)
        
        print(f"✅ Raw data loaded: {self.df.shape[0]} rows, "
              f"{self.df.shape[1]} columns")
        
        # Clean the messy data types
        self.df = self._clean_types(self.df)
        
        # Validate the data is correct
        self._validate(self.df)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total customers: {len(self.df)}")
        print(f"   Churned customers: "
              f"{self.df['Churn'].sum()} "
              f"({self.df['Churn'].mean()*100:.1f}%)")
        print(f"   Active customers: "
              f"{(self.df['Churn']==0).sum()} "
              f"({(self.df['Churn']==0).mean()*100:.1f}%)")
        print(f"   Avg Monthly Charges: "
              f"${self.df['MonthlyCharges'].mean():.2f}")
        
        return self.df
    
    def _clean_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix data type problems in the raw CSV.
        
        PROBLEM 1: TotalCharges column
        In the CSV, some rows have a space ' ' instead of a number
        in TotalCharges. This happens for new customers (tenure=0).
        pd.to_numeric with errors='coerce' converts spaces to NaN.
        We then fill those NaN values with MonthlyCharges.
        
        PROBLEM 2: Churn column
        The Churn column has 'Yes'/'No' strings.
        We convert to 1/0 numbers so ML model can understand it.
        """
        df = df.copy()  # Never modify original data
        
        # Fix TotalCharges
        print("\n🔧 Cleaning TotalCharges column...")
        
        # Count how many bad values exist before fixing
        bad_values = df['TotalCharges'].apply(
            lambda x: str(x).strip() == ''
        ).sum()
        print(f"   Found {bad_values} empty TotalCharges values")
        
        # Convert to number, spaces become NaN
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'], 
            errors='coerce'  # 'coerce' = convert errors to NaN
        )
        
        # Fill NaN with MonthlyCharges (makes business sense
        # because if tenure=0, total = first month charge)
        # NEW CODE - correct way
        df['TotalCharges'] = df['TotalCharges'].fillna(
        df['MonthlyCharges']
        )
        print(f"   ✅ Fixed {bad_values} empty values")
        
        # Fix Churn column: 'Yes' → 1, 'No' → 0
        print("\n🔧 Converting Churn column to 0/1...")
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        print(f"   ✅ Done")
        
        return df
    
    def _validate(self, df: pd.DataFrame):
        """
        Check that our data has everything we need.
        If something is wrong, stop early with a clear error message
        rather than failing silently later.
        """
        print("\n🔍 Validating data...")
        
        # Check 1: Required columns must exist
        required_cols = [
            'customerID', 'tenure', 'MonthlyCharges', 
            'TotalCharges', 'Churn', 'Contract',
            'PaymentMethod', 'InternetService'
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"❌ Missing required columns: {missing_cols}"
            )
        print(f"   ✅ All required columns present")
        
        # Check 2: No negative tenure
        if df['tenure'].min() < 0:
            raise ValueError(
                f"❌ Found negative tenure values!"
            )
        print(f"   ✅ Tenure values valid "
              f"(range: {df['tenure'].min()} - "
              f"{df['tenure'].max()} months)")
        
        # Check 3: No negative charges
        if df['MonthlyCharges'].min() < 0:
            raise ValueError(
                f"❌ Found negative MonthlyCharges!"
            )
        print(f"   ✅ MonthlyCharges valid "
              f"(range: ${df['MonthlyCharges'].min():.2f} - "
              f"${df['MonthlyCharges'].max():.2f})")
        
        # Check 4: No null values in critical columns
        null_counts = df[required_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            raise ValueError(
                f"❌ Null values found:\n{cols_with_nulls}"
            )
        print(f"   ✅ No null values in critical columns")
        
        # Check 5: Churn column only has 0 and 1
        unique_churn = set(df['Churn'].unique())
        if not unique_churn.issubset({0, 1}):
            raise ValueError(
                f"❌ Churn column has unexpected values: "
                f"{unique_churn}"
            )
        print(f"   ✅ Churn column validated")
        
        print(f"\n✅ ALL VALIDATION CHECKS PASSED!")
    
    def get_info(self):
        """Print useful information about the loaded dataset"""
        if self.df is None:
            print("❌ No data loaded yet. Call load() first.")
            return
            
        print("\n📋 Column Information:")
        print(self.df.dtypes)
        
        print("\n📋 Missing Values:")
        print(self.df.isnull().sum())
        
        print("\n📋 First 3 rows:")
        print(self.df.head(3))