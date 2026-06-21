# src/api/main.py
# ================
# FastAPI application that exposes our ML system as REST API
#
# Endpoints:
# GET  /          → Welcome message
# GET  /health    → Check if API is running
# POST /predict   → Predict churn for one customer
# POST /predict/batch → Predict for multiple customers
# GET  /model/info → Model performance metrics

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)

from src.api.schemas import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    BatchPredictionSummary
)
from src.data.preprocessing import FeatureEngineer
from src.explainability.shap_explainer import SHAPExplainer
from src.retention.strategy_engine import RetentionStrategyEngine

# ==========================================
# Initialize FastAPI App
# ==========================================
app = FastAPI(
    title="🎯 Churn Recovery Engine API",
    description="""
    ## Intelligent Customer Churn Prediction System
    
    This API predicts customer churn and automatically generates 
    personalized retention strategies.
    
    ### What it does:
    - **Predicts** churn probability using XGBoost (AUC: 0.847)
    - **Explains** WHY using SHAP values
    - **Acts** by generating AI retention strategies via LLM
    
    ### Key Features:
    - Optimized prediction threshold (0.636) for best F1 score
    - SHAP explainability for every prediction
    - LLM-powered personalized retention strategies
    - Revenue impact quantification
    """,
    version="1.0.0",
)

# Allow requests from anywhere (needed for dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Load Models at Startup
# ==========================================
# These are loaded ONCE when API starts
# Not on every request (that would be slow)

print("🔄 Loading models...")

try:
    # Load XGBoost booster
    booster = xgb.Booster()
    booster.load_model('models/churn_model.json')

    # Load supporting files
    feature_names = joblib.load('models/feature_names.pkl')
    threshold = joblib.load('models/threshold.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/encoders.pkl')
    metrics = joblib.load('models/metrics.pkl')

    # Load SHAP explainer
    shap_explainer = joblib.load('models/shap_explainer.pkl')

    # Initialize retention engine
    retention_engine = RetentionStrategyEngine()

    MODELS_LOADED = True
    print("✅ All models loaded successfully!")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODELS_LOADED = False


# ==========================================
# Helper Functions
# ==========================================

def prepare_customer_features(
    customer: CustomerInput
) -> pd.DataFrame:
    """
    Prepares customer features for prediction.
    Uses saved feature_names to ensure correct order.
    """
    customer_dict = customer.dict()
    df = pd.DataFrame([customer_dict])
    df['Churn'] = 0

    fe = FeatureEngineer()
    fe.encoders = encoders
    fe.scaler = scaler
    # ✅ Pass saved feature names so order is guaranteed
    fe.feature_names = feature_names

    df_engineered = fe.build_features(df)
    X_scaled, _ = fe.encode_and_scale(
        df_engineered,
        fit=False
    )

    # ✅ Final safety check - force correct column order
    missing_cols = set(feature_names) - set(X_scaled.columns)
    if missing_cols:
        raise ValueError(
            f"Missing features: {missing_cols}"
        )

    X_final = X_scaled[feature_names]
    return X_final


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level string"""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# ==========================================
# API Endpoints
# ==========================================

@app.get("/", tags=["General"])
def welcome():
    """Welcome endpoint"""
    return {
        "message": "🎯 Welcome to Churn Recovery Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"]
)
def health_check():
    """
    Check if API is running and models are loaded.
    Always check this first when deploying.
    """
    return HealthResponse(
        status="healthy" if MODELS_LOADED else "unhealthy",
        model_version=os.getenv('MODEL_VERSION', 'v1.0'),
        models_loaded=MODELS_LOADED,
        message=(
            "All models loaded and ready"
            if MODELS_LOADED
            else "Models failed to load"
        )
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """
    Returns model performance metrics.
    Useful for monitoring model quality over time.
    """
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )

    return {
        "model_type": "XGBoost Classifier",
        "model_version": "v1.0",
        "performance_metrics": metrics,
        "prediction_threshold": threshold,
        "feature_count": len(feature_names),
        "features_used": feature_names,
        "training_info": {
            "dataset": "Telco Customer Churn",
            "training_samples": 4929,
            "validation_samples": 705,
            "test_samples": 1409,
            "optimization": "Optuna (30 trials)",
            "explainability": "SHAP TreeExplainer"
        }
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict_churn(customer: CustomerInput):
    """
    ## Predict Churn for a Single Customer
    
    Send customer data and receive:
    - **Churn probability** (0 to 1)
    - **Risk level** (HIGH/MEDIUM/LOW)
    - **Top reasons** why they might churn (SHAP)
    - **AI retention strategy** to keep them
    - **Revenue at risk** calculation
    
    This is the main endpoint of the entire system.
    """

    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check /health endpoint."
        )

    try:
        # Step 1: Prepare features
        X = prepare_customer_features(customer)

        # Step 2: Get churn probability
        dmatrix = xgb.DMatrix(X.values)
        churn_prob = float(booster.predict(dmatrix)[0])

        # Step 3: Apply threshold
        risk_level = get_risk_level(churn_prob)
        will_churn = churn_prob >= threshold

        # Step 4: SHAP explanation
        explanation = shap_explainer.explain_single_customer(
            X,
            customer_id=customer.customerID
        )

        # Step 5: Generate retention strategy
        customer_profile = {
            'Contract': customer.Contract,
            'tenure': customer.tenure,
            'InternetService': customer.InternetService,
            'TechSupport': customer.TechSupport,
            'OnlineSecurity': customer.OnlineSecurity,
            'PaymentMethod': customer.PaymentMethod,
            'SeniorCitizen': customer.SeniorCitizen,
        }

        strategy = retention_engine.generate_strategy(
            customer_profile=customer_profile,
            churn_probability=churn_prob,
            top_risk_factors=explanation['top_risk_factors'],
            top_protective_factors=explanation[
                'top_protective_factors'
            ],
            monthly_charges=customer.MonthlyCharges
        )

        # Step 6: Build response
        annual_revenue = customer.MonthlyCharges * 12

        return PredictionResponse(
            customer_id=customer.customerID,
            churn_probability=round(churn_prob, 4),
            churn_risk_level=risk_level,
            will_churn=will_churn,
            top_risk_factors=explanation['top_risk_factors'],
            top_protective_factors=explanation[
                'top_protective_factors'
            ],
            explanation_text=explanation['explanation_text'],
            retention_strategy=strategy,
            annual_revenue_at_risk=round(annual_revenue, 2),
            model_version="v1.0"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionSummary,
    tags=["Prediction"]
)
def predict_batch(customers: list[CustomerInput]):
    """
    ## Predict Churn for Multiple Customers
    
    Send a list of customers and get predictions for all.
    Also returns a summary with total revenue at risk.
    
    Useful for processing entire customer segments.
    """

    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )

    if len(customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 customers per batch request"
        )

    predictions = []
    high_risk = 0
    medium_risk = 0
    low_risk = 0
    total_revenue_at_risk = 0

    for customer in customers:
        try:
            # Get prediction for each customer
            result = predict_churn(customer)
            predictions.append(result.dict())

            # Count risk levels
            if result.churn_risk_level == "HIGH":
                high_risk += 1
                total_revenue_at_risk += (
                    result.annual_revenue_at_risk
                )
            elif result.churn_risk_level == "MEDIUM":
                medium_risk += 1
            else:
                low_risk += 1

        except Exception as e:
            # Skip failed predictions but continue batch
            print(f"Failed for customer "
                  f"{customer.customerID}: {e}")

    return BatchPredictionSummary(
        total_customers=len(customers),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
        total_revenue_at_risk=round(total_revenue_at_risk, 2),
        predictions=predictions
    )


@app.get("/predict/sample", tags=["Prediction"])
def get_sample_prediction():
    """
    ## Get a Sample Prediction
    
    Returns a prediction for a pre-filled high-risk customer.
    Great for testing the API without building a frontend.
    """
    sample_customer = CustomerInput(
        customerID="SAMPLE-HIGH-RISK",
        tenure=2,
        MonthlyCharges=85.5,
        TotalCharges=171.0,
        Contract="Month-to-month",
        PaymentMethod="Electronic check",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        PhoneService="Yes",
        MultipleLines="No",
        OnlineBackup="No",
        DeviceProtection="No",
        gender="Male",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        PaperlessBilling="Yes"
    )

    return predict_churn(sample_customer)