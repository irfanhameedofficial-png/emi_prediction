import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
from sklearn.preprocessing import LabelEncoder

# GOOGLE DRIVE FILE IDs
#4GB Model
# REG_MODEL_ID = "1Uc3cEFXdM-F0M3YNmE6wwokxdnbbdxdq"
# CLS_MODEL_ID = "1EnMQLiWiGbbOc7xfNKEX2qlP2ldGrQ7L"
#1st Compression Model 
# REG_MODEL_ID = "1DxtLMzApBMp40bIybvI7W0e-6eHCHcdD"
# CLS_MODEL_ID = "1acxjnD5MC6vzt_xSsDlvSwrPjwhKaK_p"
# 2nd Compression Model using lzma
REG_MODEL_ID = "1qBR60SfO4NlFQkN9YDb6RFI7_zEjd33z"
CLS_MODEL_ID = "10IsgfjZWN641uO4DLBWMsJN3Za4oBqPk"
#Lightweight Model for Reg
# REG_MODEL_ID = "1Vx4kuP2t657XjeRkXMxWcQWwxo9h_WeN"

CSV_FILE_ID = "1Gvw7MMF0jLSseRC4EP9rkPAUbEiP6v3K"  

# LOCAL FILE PATHS
REG_MODEL_PATH = "Final_Regression_compressed.pkl"
# REG_MODEL_PATH = "Regression2.pkl"
CLS_MODEL_PATH = "Final_Classification_compressed.pkl"
CSV_PATH = "cleaned_emi_dataset.csv"

# DOWNLOAD FUNCTION
def download_files():
    st.info("📦 Downloading required files from Google Drive...")
    if not os.path.exists(REG_MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={REG_MODEL_ID}", REG_MODEL_PATH, quiet=False)
    if not os.path.exists(CLS_MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={CLS_MODEL_ID}", CLS_MODEL_PATH, quiet=False)
    if not os.path.exists(CSV_PATH):
        gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", CSV_PATH, quiet=False)
    st.success("✅ All files downloaded successfully!")

# LOAD DATA
if not os.path.exists(CSV_PATH):
    download_files()

df = pd.read_csv(CSV_PATH)

# Calculate mode and median for backend filling
mode_values = df.mode().iloc[0]
median_values = df.median(numeric_only=True)

# UI TITLE
st.title("🏦 EMI Prediction App")
st.markdown("Enter your details below to check **EMI eligibility** and **predicted EMI amount**.")

# USER INPUT FORM
with st.form("emi_form"):
    st.header("🔸 Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_salary = st.number_input("Monthly Salary", min_value=0.0)
        house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
        school_fees = st.number_input("School Fees", min_value=0.0)
        college_fees = st.number_input("College Fees", min_value=0.0)
        
    with col2:
        groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0)
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
        current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0)
        credit_score = st.number_input("Credit Score", min_value=0.0)

    with col3:
        requested_amount = st.number_input("Requested Loan Amount", min_value=0.0)        
        bank_balance = st.number_input("Bank Balance", min_value=0.0)
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, step=1)

    submitted = st.form_submit_button("🔍 Predict")

# PREDICTION LOGIC
if submitted:
    st.info("⏳ Processing your input...")

    # Ensure model files exist
    if not (os.path.exists(REG_MODEL_PATH) and os.path.exists(CLS_MODEL_PATH)):
        download_files()

    # Load models
    reg = joblib.load(REG_MODEL_PATH)
    cls = joblib.load(CLS_MODEL_PATH)

    # Create base input with median/mode defaults
    input_data = pd.DataFrame([{
        col: (median_values[col] if col in median_values else mode_values[col])
        for col in df.columns if col not in ['emi_eligibility', 'max_monthly_emi']
    }])

    # Update with user-entered values
    input_data['monthly_salary'] = monthly_salary
    input_data['house_type'] = house_type
    input_data['college_fees'] = college_fees
    input_data['current_emi_amount'] = current_emi_amount
    input_data['school_fees'] = school_fees
    input_data['existing_loans'] = existing_loans
    input_data['credit_score'] = credit_score
    input_data['bank_balance'] = bank_balance
    input_data['groceries_utilities'] = groceries_utilities
    input_data['requested_amount'] = requested_amount
    input_data['requested_tenure'] = requested_tenure

    # Label encode categorical columns
    cat_cols = ['gender', 'marital_status', 'education', 'employment_type',
                'company_type', 'house_type', 'existing_loans', 'emi_scenario']

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        input_data[col] = le.transform(input_data[col].astype(str))
        label_encoders[col] = le

    # ✅ Check encoding of emi_eligibility (for verification only)
    # emi_le = LabelEncoder()
    # emi_le.fit(df['emi_eligibility'].astype(str))
    # encoded_labels = dict(zip(emi_le.classes_, emi_le.transform(emi_le.classes_)))

    # st.info("🧩 EMI Eligibility Encoding Check:")
    # st.write(pd.DataFrame(encoded_labels.items(), columns=["Category", "Encoded Value"]))

    # Predict Eligibility
    eligibility_pred = cls.predict(input_data)[0]
    eligibility_map = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
    eligibility_status = eligibility_map.get(eligibility_pred, "Unknown")

    # Predict EMI amount
    emi_pred = reg.predict(input_data)[0]

    # Display results
    st.success(f"✅ **Eligibility:** {eligibility_status}")
    st.success(f"💰 **Predicted EMI Amount:** ₹{emi_pred:,.2f}")
