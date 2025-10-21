import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# ============================
# 🔹 GOOGLE DRIVE FILE IDs
# ============================
REG_MODEL_ID = "1Vx4kuP2t657XjeRkXMxWcQWwxo9h_WeN"
CLS_MODEL_ID = "1ygDCeb7J69y6M0Gt-S0RJjRYjsXY4oPx"
CSV_FILE_ID = "1Gvw7MMF0jLSseRC4EP9rkPAUbEiP6v3K"   # 🔁 Replace with your actual dataset file ID

# ============================
# 🔹 LOCAL FILE PATHS
# ============================
REG_MODEL_PATH = "Regression.pkl"
CLS_MODEL_PATH = "Classification.pkl"
CSV_PATH = "cleaned_emi_dataset.csv"

# ============================
# 🔹 DOWNLOAD FUNCTION
# ============================
def download_files():
    st.info("📦 Downloading required files from Google Drive...")
    if not os.path.exists(REG_MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={REG_MODEL_ID}", REG_MODEL_PATH, quiet=False)
    if not os.path.exists(CLS_MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={CLS_MODEL_ID}", CLS_MODEL_PATH, quiet=False)
    if not os.path.exists(CSV_PATH):
        gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", CSV_PATH, quiet=False)
    st.success("✅ All files downloaded successfully!")

# ============================
# 🔹 LOAD DATA (for encoders)
# ============================
if not os.path.exists(CSV_PATH):
    download_files()

df = pd.read_csv(CSV_PATH)

# ============================
# 🔹 UI TITLE
# ============================
st.title("🏦 EMI Prediction App")
st.markdown("Enter your details below to check **EMI eligibility** and **predicted EMI amount**.")

# ============================
# 🔹 USER INPUT FORM
# ============================
with st.form("emi_form"):
    st.header("🔸 Enter Applicant Details")

    # Numeric Inputs
    age = st.number_input("Age", min_value=18, max_value=70, step=1)
    monthly_salary = st.number_input("Monthly Salary", min_value=0.0)
    years_of_employment = st.number_input("Years of Employment", min_value=0.0)
    monthly_rent = st.number_input("Monthly Rent", min_value=0.0)
    family_size = st.number_input("Family Size", min_value=1, step=1)
    dependents = st.number_input("Dependents", min_value=0, step=1)
    school_fees = st.number_input("School Fees", min_value=0.0)
    college_fees = st.number_input("College Fees", min_value=0.0)
    travel_expenses = st.number_input("Travel Expenses", min_value=0.0)
    groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0.0)
    current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=0.0)
    bank_balance = st.number_input("Bank Balance", min_value=0.0)
    emergency_fund = st.number_input("Emergency Fund", min_value=0.0)
    requested_amount = st.number_input("Requested Loan Amount", min_value=0.0)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, step=1)

    # Categorical Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education", ["Graduate", "Post Graduate", "High School", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    company_type = st.selectbox("Company Type", ["Large Indian", "MNC", "Mid-size", "Startup", "Small"])
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    emi_scenario = st.selectbox("EMI Scenario", ["Home Appliances EMI", "Personal Loan EMI", "E-commerce Shopping EMI", "Education EMI", "Vehicle EMI"])

    submitted = st.form_submit_button("🔍 Predict")

# ============================
# 🔹 PREDICTION LOGIC
# ============================
if submitted:
    st.info("⏳ Processing your input...")

    # Ensure model files exist
    if not (os.path.exists(REG_MODEL_PATH) and os.path.exists(CLS_MODEL_PATH)):
        download_files()

    # Load models
    rf_reg = joblib.load(REG_MODEL_PATH)
    xgb_cls = joblib.load(CLS_MODEL_PATH)

    # Create DataFrame from inputs
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'marital_status': marital_status,
        'education': education,
        'monthly_salary': monthly_salary,
        'employment_type': employment_type,
        'years_of_employment': years_of_employment,
        'company_type': company_type,
        'house_type': house_type,
        'monthly_rent': monthly_rent,
        'family_size': family_size,
        'dependents': dependents,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_expenses,
        'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,
        'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount,
        'credit_score': credit_score,
        'bank_balance': bank_balance,
        'emergency_fund': emergency_fund,
        'emi_scenario': emi_scenario,
        'requested_amount': requested_amount,
        'requested_tenure': requested_tenure
    }])

    # Label encode categorical columns
    cat_cols = ['gender', 'marital_status', 'education', 'employment_type',
                'company_type', 'house_type', 'existing_loans', 'emi_scenario']

    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        input_data[col] = le.transform(input_data[col].astype(str))
        label_encoders[col] = le

    # Predict Eligibility
    eligibility_pred = xgb_cls.predict(input_data)[0]
    eligibility_map = {0: "Not_Eligible", 1: "Eligible", 2: "High_Risk"}
    eligibility_status = eligibility_map.get(eligibility_pred, "Unknown")

    # Predict EMI amount
    emi_pred = rf_reg.predict(input_data)[0]

    # Display results
    st.success(f"✅ **Eligibility:** {eligibility_status}")
    st.success(f"💰 **Predicted EMI Amount:** ₹{emi_pred:,.2f}")
