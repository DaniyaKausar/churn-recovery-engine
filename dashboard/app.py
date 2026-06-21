# dashboard/app.py
# =================
# Streamlit dashboard for Churn Recovery Engine
#
# This is the visual interface for our entire ML system.
# It calls the FastAPI backend and displays results beautifully.
#
# Run with: streamlit run dashboard/app.py

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Churn Recovery Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS for better styling
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .high-risk {
        background-color: #ffe0e0;
        border-left: 5px solid #ff4444;
        padding: 1rem;
        border-radius: 5px;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Header
# ==========================================
st.markdown(
    '<div class="main-header">🎯 Churn Recovery Engine</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">'
    'Predict → Explain → Retain | '
    'XGBoost + SHAP + LLM</div>',
    unsafe_allow_html=True
)

# API URL
import os

# Use production URL if deployed, otherwise localhost for testing
API_URL = os.environ.get(
    "API_URL", 
    "https://churn-recovery-engine-production.up.railway.app"
)
# ==========================================
# Sidebar — Customer Input Form
# ==========================================
st.sidebar.header("👤 Customer Details")
st.sidebar.markdown("Fill in customer information to predict churn risk")

# Contract & Billing
st.sidebar.subheader("📋 Contract Info")
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    help="Month-to-month customers have highest churn risk"
)

tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=6,
    help="How long has the customer been with you?"
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=20.0,
    max_value=120.0,
    value=75.0,
    step=0.5
)

total_charges = monthly_charges * max(tenure, 1)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

# Services
st.sidebar.subheader("🌐 Services")
internet = st.sidebar.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

# Demographics
st.sidebar.subheader("👥 Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)
senior_int = 1 if senior == "Yes" else 0

partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox(
    "Has Dependents",
    ["No", "Yes"]
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

# Customer ID
customer_id = st.sidebar.text_input(
    "Customer ID",
    value="CUST-001"
)

# ==========================================
# Predict Button
# ==========================================
predict_button = st.sidebar.button(
    "🔍 Analyze Customer",
    type="primary",
    use_container_width=True
)

# ==========================================
# Check API Health
# ==========================================
def check_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False

# ==========================================
# Main Content
# ==========================================

if not predict_button:
    # Show welcome screen
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 ML Model
        - **XGBoost** with Optuna tuning
        - **AUC-ROC:** 0.8459
        - **F1 Score:** 0.6301
        - **29 engineered features**
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Explainability  
        - **SHAP** TreeExplainer
        - Per-customer explanations
        - Top risk & protective factors
        - Business-readable insights
        """)
    
    with col3:
        st.markdown("""
        ### 💡 AI Strategy
        - **LLM-powered** retention plans
        - Personalized offers
        - Revenue impact calculation
        - Action deadlines & channels
        """)
    
    st.markdown("---")
    st.info(
        "👈 Fill in customer details in the sidebar "
        "and click **Analyze Customer** to get started!"
    )
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API is running and ready")
    else:
        st.error(
            "❌ API is not running. "
            "Please start the API with: python run_step7.py"
        )

else:
    # Check if API is running
    if not check_api_health():
        st.error(
            "❌ Cannot connect to API! "
            "Make sure run_step7.py is running in another terminal."
        )
        st.stop()
    
    # Build request payload
    payload = {
        "customerID": customer_id,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "gender": gender,
        "SeniorCitizen": senior_int,
        "Partner": partner,
        "Dependents": dependents,
        "PaperlessBilling": paperless
    }
    
    # Call API
    with st.spinner("🤖 Analyzing customer..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            result = response.json()
        except Exception as e:
            st.error(f"❌ API call failed: {e}")
            st.stop()
    
    if response.status_code != 200:
        st.error(f"❌ API Error: {result.get('detail', 'Unknown error')}")
        st.stop()
    
    # ==========================================
    # Display Results
    # ==========================================
    
    churn_prob = result['churn_probability']
    risk_level = result['churn_risk_level']
    will_churn = result['will_churn']
    strategy = result['retention_strategy']
    annual_revenue = result['annual_revenue_at_risk']
    
    # Risk level styling
    risk_colors = {
        "HIGH": "#ff4444",
        "MEDIUM": "#ff9800",
        "LOW": "#4caf50"
    }
    risk_color = risk_colors.get(risk_level, "#666")
    
    # ── Row 1: Key Metrics ──
    st.markdown("## 📊 Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_prob:.1%}",
            delta=f"Risk: {risk_level}"
        )
    
    with col2:
        st.metric(
            label="Will Churn?",
            value="YES ⚠️" if will_churn else "NO ✅",
            delta=f"Threshold: {0.636}"
        )
    
    with col3:
        st.metric(
            label="Annual Revenue at Risk",
            value=f"${annual_revenue:,.0f}",
            delta="per year"
        )
    
    with col4:
        revenue_saved = strategy.get(
            'estimated_revenue_saved', 0
        )
        st.metric(
            label="Revenue Saved (if retained)",
            value=f"${revenue_saved:,.0f}",
            delta="with action"
        )
    
    st.markdown("---")
    
    # ── Row 2: Gauge + SHAP Chart ──
    col_gauge, col_shap = st.columns(2)
    
    with col_gauge:
        st.markdown("### 🎯 Churn Risk Gauge")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': "Churn Risk %",
                'font': {'size': 18}
            },
            delta={
                'reference': 26.5,
                'increasing': {'color': "red"},
                'decreasing': {'color': "green"}
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1
                },
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'steps': [
                    {
                        'range': [0, 40],
                        'color': '#e8f5e9'
                    },
                    {
                        'range': [40, 70],
                        'color': '#fff3e0'
                    },
                    {
                        'range': [70, 100],
                        'color': '#ffe0e0'
                    }
                ],
                'threshold': {
                    'line': {
                        'color': "black",
                        'width': 4
                    },
                    'thickness': 0.75,
                    'value': 63.6  # Our optimal threshold
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Delta explanation
        st.caption(
            "📌 Black line = decision threshold (63.6%). "
            "Delta = vs average churn rate (26.5%)"
        )
    
    with col_shap:
        st.markdown("### 🔍 Top Churn Drivers (SHAP)")
        
        risk_factors = result.get('top_risk_factors', [])
        protective_factors = result.get(
            'top_protective_factors', []
        )
        
        # Combine for chart
        all_factors = []
        
        for f in risk_factors[:4]:
            all_factors.append({
                'feature': f['feature'],
                'impact': f['shap_value'],
                'type': 'Risk Factor'
            })
        
        for f in protective_factors[:3]:
            all_factors.append({
                'feature': f['feature'],
                'impact': f['shap_value'],
                'type': 'Protective'
            })
        
        if all_factors:
            factors_df = pd.DataFrame(all_factors)
            factors_df = factors_df.sort_values(
                'impact', ascending=True
            )
            
            colors = [
                '#ff4444' if v > 0 else '#4caf50'
                for v in factors_df['impact']
            ]
            
            fig_shap = go.Figure(go.Bar(
                x=factors_df['impact'],
                y=factors_df['feature'],
                orientation='h',
                marker_color=colors,
                text=[
                    f"+{v:.3f}" if v > 0 else f"{v:.3f}"
                    for v in factors_df['impact']
                ],
                textposition='outside'
            ))
            
            fig_shap.update_layout(
                height=300,
                margin=dict(l=20, r=60, t=20, b=20),
                xaxis_title="SHAP Impact",
                showlegend=False,
                xaxis=dict(
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='black'
                )
            )
            
            st.plotly_chart(
                fig_shap,
                use_container_width=True
            )
            st.caption(
                "🔴 Red = increases churn risk  "
                "🟢 Green = reduces churn risk"
            )
    
    st.markdown("---")
    
    # ── Row 3: SHAP Explanation + Strategy ──
    col_explain, col_strategy = st.columns(2)
    
    with col_explain:
        st.markdown("### 📋 Why This Prediction?")
        
        explanation = result.get('explanation_text', '')
        
        if risk_factors:
            st.markdown("**⚠️ Churn Risk Factors:**")
            for f in risk_factors[:4]:
                impact_pct = f['shap_value'] * 100
                st.markdown(
                    f"- **{f['feature']}**: "
                    f"+{impact_pct:.1f}% churn push"
                )
        
        if protective_factors:
            st.markdown("\n**✅ Protective Factors:**")
            for f in protective_factors[:3]:
                impact_pct = abs(f['shap_value']) * 100
                st.markdown(
                    f"- **{f['feature']}**: "
                    f"-{impact_pct:.1f}% protection"
                )
        
        # Customer summary
        st.markdown("\n**👤 Customer Summary:**")
        st.markdown(f"- Contract: **{contract}**")
        st.markdown(f"- Tenure: **{tenure} months**")
        st.markdown(
            f"- Monthly Charges: **${monthly_charges:.2f}**"
        )
        st.markdown(f"- Internet: **{internet}**")
    
    with col_strategy:
        st.markdown("### 💡 AI Retention Strategy")
        
        urgency = strategy.get('urgency_level', 'MEDIUM')
        
        # Urgency badge
        urgency_colors = {
            "HIGH": "🔴",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }
        badge = urgency_colors.get(urgency, "⚪")
        
        st.markdown(
            f"**{badge} Urgency: {urgency}** | "
            f"Act within **"
            f"{strategy.get('action_deadline_days', 7)}"
            f" days** | "
            f"Channel: **{strategy.get('channel', 'Email')}**"
        )
        
        st.info(
            f"**Primary Action:**\n"
            f"{strategy.get('primary_action', 'N/A')}"
        )
        
        st.warning(
            f"**Offer:**\n"
            f"{strategy.get('offer', 'N/A')}"
        )
        
        talking_points = strategy.get('talking_points', [])
        if talking_points:
            st.markdown("**Talking Points:**")
            for i, point in enumerate(talking_points, 1):
                st.markdown(f"{i}. {point}")
        
        # Outcome metrics
        ret_prob = strategy.get(
            'estimated_retention_probability', 0
        )
        
        st.success(
            f"**Expected Outcome:** "
            f"{ret_prob:.0%} retention probability | "
            f"${revenue_saved:,.0f} revenue saved"
        )
        
        reasoning = strategy.get('reasoning', '')
        if reasoning:
            st.caption(f"💭 {reasoning}")
    
    st.markdown("---")
    
    # ── Row 4: Raw JSON (for developers) ──
    with st.expander(
        "🔧 View Raw API Response (for developers)"
    ):
        st.json(result)
    
    # ── Footer ──
    st.markdown("---")
    st.caption(
        "Built with XGBoost + SHAP + LLM | "
        "Model AUC: 0.8459 | "
        "Threshold: 0.636 | "
        "Features: 29"
    )
    st.success(f"**Primary Action:** {strategy.get('primary_action')}")
    st.warning(f"**Offer:** {strategy.get('offer')}")
    
    st.markdown("**Talking Points:**")
    for point in strategy.get('talking_points', []):
        st.markdown(f"- {point}")
