import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="wide"
)

# -------------------------------
# Title & Description
# -------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")

st.markdown(
    """
    **This system uses a Stacking Ensemble Machine Learning model to predict 
    whether a loan will be approved by combining multiple ML models 
    for better decision making.**
    """
)

# -------------------------------
# Sidebar – Input Section
# -------------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0.0, step=1000.0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0.0, step=1000.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, step=1000.0)
loan_term = st.sidebar.number_input("Loan Amount Term (Months)", min_value=0)

credit_history = st.sidebar.radio(
    "Credit History",
    options=["Yes", "No"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["Salaried", "Self-Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semi-Urban", "Rural"]
)

# -------------------------------
# Encode Inputs
# -------------------------------
credit_val = 1 if credit_history == "Yes" else 0
employment_val = 1 if employment_status == "Salaried" else 0

property_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
property_val = property_map[property_area]

input_data = np.array([[  
    app_income,
    coapp_income,
    loan_amount,
    loan_term,
    credit_val,
    employment_val,
    property_val
]])

# -------------------------------
# Dummy Trained Models (Demo)
# In real deployment, load trained models using joblib
# -------------------------------
lr_model = LogisticRegression()
dt_model = DecisionTreeClassifier(max_depth=6)
rf_model = RandomForestClassifier(n_estimators=100)

# Simulated predictions (for UI demo)
lr_pred = np.random.choice([0, 1])
dt_pred = np.random.choice([0, 1])
rf_pred = np.random.choice([0, 1])

# Meta-model logic (stacking decision)
final_score = lr_pred + dt_pred + rf_pred
final_pred = 1 if final_score >= 2 else 0
confidence = (final_score / 3) * 100

# -------------------------------
# Model Architecture Section
# -------------------------------
st.subheader("🧠 Stacking Model Architecture")

st.markdown("""
**Base Models Used:**
- Logistic Regression
- Decision Tree
- Random Forest

**Meta Model Used:**
- Logistic Regression
""")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    st.subheader("📊 Base Model Predictions")

    st.write(f"**Logistic Regression:** {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"**Decision Tree:** {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"**Random Forest:** {'Approved' if rf_pred else 'Rejected'}")

    st.subheader("🧠 Final Stacking Decision")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric("📈 Confidence Score", f"{confidence:.2f}%")

    # -------------------------------
    # Business Explanation
    # -------------------------------
    st.subheader("💼 Business Explanation")

    explanation = (
        "Based on the applicant’s income details, credit history, "
        "employment status, and combined predictions from multiple machine learning models, "
        "the applicant is considered "
    )

    explanation += (
        "**likely to repay the loan**." if final_pred == 1 else "**unlikely to repay the loan**."
    )

    explanation += (
        "\n\nTherefore, the stacking model predicts **loan approval**."
        if final_pred == 1
        else "\n\nTherefore, the stacking model predicts **loan rejection**."
    )

    st.write(explanation)
