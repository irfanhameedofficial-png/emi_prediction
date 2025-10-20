import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Load dataset for column reference ---
df = pd.read_csv("cleaned_emi_dataset.csv")

# --- Load trained models ---
xgb_cls = joblib.load(r"C:\emi_prediction\Classification.pkl")  # Classification model
rf_reg = joblib.load(r"C:\emi_prediction\Regression.pkl")       # Regression model

# --- Label Encoding (as used during training) ---
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- Streamlit Config ---
st.set_page_config(page_title="EMI Prediction App", layout="centered")
st.title("💰 EMI Prediction & Eligibility Checker")

st.markdown("""
Welcome to the **EMI Prediction App** 👋  
This tool predicts:
- 💸 Your **maximum affordable EMI**
- ✅ Your **loan eligibility status**

Fill in your details below and click **Predict**.
""")

# --- Collect readable user inputs ---
st.header("📋 Enter Your Details")

col1, col2 = st.columns(2)

input_data = {}

# Example readable mapping — adjust based on your dataset columns
with col1:
    input_data["age"] = st.slider("Age", 18, 70, 30)
    input_data["monthly_income"] = st.number_input("Monthly Income (₹)", min_value=1000.0, max_value=1000000.0, value=50000.0)
    input_data["existing_loans"] = st.number_input("Number of Existing Loans", min_value=0, max_value=10, value=1)
    input_data["credit_score"] = st.slider("Credit Score", 300, 900, 750)
    input_data["dependents"] = st.selectbox("Number of Dependents", options=[0, 1, 2, 3, "3+"])
    
with col2:
    input_data["employment_type"] = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed", "Student"])
    input_data["house_type"] = st.selectbox("House Type", ["Owned", "Rented", "Mortgaged", "Living with Family"])
    input_data["location"] = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"])
    input_data["monthly_expenses"] = st.number_input("Monthly Expenses (₹)", min_value=0.0, max_value=1000000.0, value=20000.0)
    input_data["loan_tenure"] = st.slider("Desired Loan Tenure (Months)", 6, 120, 36)

# --- Convert to DataFrame ---
input_df = pd.DataFrame([input_data])

# --- Encode categorical columns like training ---
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# --- Align columns ---
expected_features = df.drop(["emi_eligibility", "max_monthly_emi"], axis=1).columns
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# --- Predict Button ---
if st.button("🔮 Predict EMI & Eligibility"):
    # Predict EMI (Regression)
    predicted_emi = rf_reg.predict(input_df)[0]

    # Predict Eligibility (Classification)
    predicted_eligibility = xgb_cls.predict(input_df)[0]

    # Decode eligibility label
    if "emi_eligibility" in label_encoders:
        predicted_eligibility = label_encoders["emi_eligibility"].inverse_transform([int(predicted_eligibility)])[0]

    # --- Display Results ---
    st.success("✅ Prediction Completed Successfully!")
    st.subheader("📊 Prediction Results")
    st.metric("Predicted Max EMI Amount (₹)", f"{predicted_emi:,.2f}")
    st.metric("EMI Eligibility Status", predicted_eligibility)
