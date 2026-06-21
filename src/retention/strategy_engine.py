# src/retention/strategy_engine.py
# ==================================
# This file uses GPT to generate personalized
# retention strategies based on ML predictions
#
# What it does:
# 1. Takes customer profile + churn probability + SHAP reasons
# 2. Sends all this to GPT with a smart prompt
# 3. GPT returns specific retention actions
# 4. We parse and return structured strategy

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load .env file


class RetentionStrategyEngine:
    """
    Uses GPT to generate personalized retention strategies.
    
    Why LLM instead of rule-based?
    Rule-based: "If month-to-month → offer annual plan"
    LLM-based: Considers ALL factors together and generates
    nuanced, contextual strategies like a real CSM would.
    
    This is the bridge between ML predictions and 
    real business action.
    """
    
    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            raise ValueError(
                "❌ GROQ_API_KEY not found in .env file!\n"
                "Add: GROQ_API_KEY=sk-your-key-here"
            )
        
        self.client = Groq(api_key=api_key)
        print("✅ RetentionStrategyEngine initialized")
    
    def generate_strategy(
        self,
        customer_profile: dict,
        churn_probability: float,
        top_risk_factors: list,
        top_protective_factors: list,
        monthly_charges: float
    ) -> dict:
        """
        Generates personalized retention strategy for one customer.
        
        Returns structured dict with:
        - urgency_level: HIGH/MEDIUM/LOW
        - primary_action: What to do immediately
        - offer: Specific discount or upgrade to offer
        - talking_points: What to say to the customer
        - estimated_retention_probability: If we act, will they stay?
        - estimated_revenue_saved: Money recovered if retained
        - action_deadline_days: How quickly to act
        - channel: Email/Call/SMS - best way to reach them
        """
        
        # Calculate revenue at risk
        annual_revenue = monthly_charges * 12
        
        # Determine urgency based on probability
        if churn_probability >= 0.7:
            urgency = "HIGH"
        elif churn_probability >= 0.4:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        # Format risk factors for prompt
        risk_text = "\n".join([
            f"  - {f['feature']}: "
            f"+{f['shap_value']*100:.1f}% churn push"
            for f in top_risk_factors[:3]
        ]) if top_risk_factors else "  - No major risk factors"
        
        # Format protective factors
        protect_text = "\n".join([
            f"  - {f['feature']}: "
            f"-{abs(f['shap_value'])*100:.1f}% protection"
            for f in top_protective_factors[:3]
        ]) if top_protective_factors else "  - No protective factors"
        
        # Build the prompt
        prompt = f"""
You are a senior customer success manager at a telecom company.
Analyze this at-risk customer and create a specific retention strategy.

CUSTOMER ANALYSIS:
==================
Churn Probability: {churn_probability:.1%}
Urgency Level: {urgency}
Annual Revenue at Risk: ${annual_revenue:.0f}
Monthly Charges: ${monthly_charges:.2f}
Contract Type: {customer_profile.get('Contract', 'Unknown')}
Tenure: {customer_profile.get('tenure', 0)} months
Internet Service: {customer_profile.get('InternetService', 'Unknown')}
Tech Support: {customer_profile.get('TechSupport', 'Unknown')}
Online Security: {customer_profile.get('OnlineSecurity', 'Unknown')}
Payment Method: {customer_profile.get('PaymentMethod', 'Unknown')}
Senior Citizen: {'Yes' if customer_profile.get('SeniorCitizen', 0) == 1 else 'No'}

TOP CHURN RISK FACTORS (from ML model):
{risk_text}

PROTECTIVE FACTORS (what's keeping them):
{protect_text}

TASK:
Create a specific, actionable retention strategy.
Consider the customer's tenure, charges, contract type,
and the specific reasons they are likely to churn.

Respond ONLY with valid JSON in this exact format:
{{
    "urgency_level": "HIGH or MEDIUM or LOW",
    "primary_action": "One specific immediate action",
    "offer": "Specific discount percentage or feature upgrade",
    "talking_points": [
        "Point 1 addressing their specific concern",
        "Point 2 highlighting value they get",
        "Point 3 about the specific offer"
    ],
    "channel": "Call or Email or SMS",
    "best_time_to_contact": "Morning or Afternoon or Evening",
    "estimated_retention_probability": 0.0,
    "estimated_revenue_saved": 0,
    "action_deadline_days": 0,
    "reasoning": "Brief explanation of strategy choice"
}}

Be specific to this customer's situation.
Return ONLY the JSON, no other text.
"""
        
        try:
            print(" Calling Groq for retention strategy...")
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a customer retention expert. "
                            "Always respond with valid JSON only."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower = more consistent
                max_tokens=500
            )
            
            # Extract the response text
            response_text = (
                response.choices[0].message.content.strip()
            )
            
            # Clean up response if needed
            # Sometimes GPT adds ```json ``` markdown
            if response_text.startswith('```'):
                response_text = response_text.split(
                    '```'
                )[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            # Parse JSON
            strategy = json.loads(response_text)
            
            # Add calculated fields
            strategy['annual_revenue_at_risk'] = round(
                annual_revenue, 2
            )
            strategy['churn_probability'] = round(
                churn_probability, 4
            )
            strategy['monthly_charges'] = monthly_charges
            
            return strategy
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON parsing failed: {e}")
            print(f"   Raw response: {response_text[:200]}")
            # Return fallback strategy
            return self._fallback_strategy(
                churn_probability,
                annual_revenue,
                urgency,
                monthly_charges
            )
            
        except Exception as e:
            print(f"   ⚠️  Groq call failed: {e}")
            return self._fallback_strategy(
                churn_probability,
                annual_revenue,
                urgency,
                monthly_charges
            )
    
    def _fallback_strategy(
        self,
        churn_probability: float,
        annual_revenue: float,
        urgency: str,
        monthly_charges: float
    ) -> dict:
        """
        Returns a rule-based strategy if Groq call fails.
        Ensures the system always returns something useful.
        """
        
        if urgency == "HIGH":
            return {
                "urgency_level": "HIGH",
                "primary_action": (
                    "Call customer immediately with "
                    "personalized retention offer"
                ),
                "offer": (
                    "20% discount for 6 months + "
                    "free tech support upgrade"
                ),
                "talking_points": [
                    "We value your loyalty and want to keep you",
                    "We have a special offer just for you",
                    "Let us resolve any issues you're facing"
                ],
                "channel": "Call",
                "best_time_to_contact": "Morning",
                "estimated_retention_probability": 0.65,
                "estimated_revenue_saved": round(
                    annual_revenue * 0.65, 2
                ),
                "action_deadline_days": 3,
                "annual_revenue_at_risk": annual_revenue,
                "churn_probability": churn_probability,
                "monthly_charges": monthly_charges,
                "reasoning": "High risk customer needs immediate personal outreach"
            }
        else:
            return {
                "urgency_level": urgency,
                "primary_action": (
                    "Send personalized email with upgrade offer"
                ),
                "offer": "10% discount on annual plan upgrade",
                "talking_points": [
                    "Thank you for being a valued customer",
                    "We have an exclusive offer for you",
                    "Upgrade for better value and savings"
                ],
                "channel": "Email",
                "best_time_to_contact": "Afternoon",
                "estimated_retention_probability": 0.45,
                "estimated_revenue_saved": round(
                    annual_revenue * 0.45, 2
                ),
                "action_deadline_days": 7,
                "annual_revenue_at_risk": annual_revenue,
                "churn_probability": churn_probability,
                "monthly_charges": monthly_charges,
                "reasoning": "Medium risk customer needs targeted offer"
            }
    
    def generate_batch_strategies(
        self,
        customers_data: list
    ) -> list:
        """
        Generate strategies for multiple customers at once.
        Used for batch processing in the API.
        """
        strategies = []
        
        for i, customer in enumerate(customers_data):
            print(f"   Processing customer "
                  f"{i+1}/{len(customers_data)}...")
            
            strategy = self.generate_strategy(
                customer_profile=customer['profile'],
                churn_probability=customer['churn_probability'],
                top_risk_factors=customer['risk_factors'],
                top_protective_factors=customer[
                    'protective_factors'
                ],
                monthly_charges=customer['monthly_charges']
            )
            strategies.append(strategy)
        
        return strategies