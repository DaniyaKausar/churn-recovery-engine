# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List


class CustomerInput(BaseModel):
    customerID: str = "CUST-001"
    tenure: int = 2
    MonthlyCharges: float = 85.5
    TotalCharges: float = 171.0
    Contract: str = "Month-to-month"
    PaymentMethod: str = "Electronic check"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PaperlessBilling: str = "Yes"


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk_level: str
    will_churn: bool
    top_risk_factors: list
    top_protective_factors: list
    explanation_text: str
    retention_strategy: dict
    annual_revenue_at_risk: float
    model_version: str


class BatchPredictionSummary(BaseModel):
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    total_revenue_at_risk: float
    predictions: list


class HealthResponse(BaseModel):
    status: str
    model_version: str
    models_loaded: bool
    message: str