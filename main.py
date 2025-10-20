import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Load dataset for metadata only (to populate form) ---
df = pd.read_csv("cleaned_emi_dataset.csv")

# --- Streamlit App ---
st.set_page_config(page_title="EMI Prediction App", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "🧾 Predict EMI & Eligibility"])

# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("💰 EMI Prediction and Eligibility App")
    st.markdown("""
    This app predicts:
    - **Maximum EMI Amount** you can afford (Regression Model)
    - **EMI Eligibility Status** (Classification Model)

    The prediction is based on various financial and demographic factors such as salary, house type, loans, and expenses.

    **How it works:**
    1. Go to the *Predict EMI & Eligibility* page.
    2. Fill in your details.
    3. Click **Predict** to see both your **Eligible EMI Amount** and **Eligibility Status**.
    """)

# --- PREDICTION PAGE ---
elif page == "🧾 Predict EMI & Eligibility":
    st.title("📊 Predict Your EMI and Eligibility")
    st.markdown("Please fill in all the details below:")

    # Collect user inputs
    input_data = {}
    for col in df.drop(['emi_eligibility', 'max_monthly_emi'], axis=1).columns:
        if df[col].dtype == 'object':
            input_data[col] = st.selectbox(f"{col}", df[col].unique(), key=col)
        else:
            input_data[col] = st.number_input(f"{col}", value=0.0, key=col)

    # --- When Predict button is clicked ---
    if st.button("🔮 Predict EMI & Eligibility"):
        # --- Convert user input to DataFrame ---
        input_df = pd.DataFrame([input_data])

        # --- Encode categorical columns ---
        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))  # fit on training data only
            label_encoders[col] = le
            if col in input_df.columns:
                val = input_df.at[0, col]
                try:
                    input_df.at[0, col] = le.transform([val])[0]
                except ValueError:
                    # If unseen value, use mode
                    input_df.at[0, col] = le.transform([df[col].mode()[0]])[0]

        # --- Reorder columns to match training ---
        all_features = [
            'age', 'gender', 'marital_status', 'education', 'monthly_salary',
            'employment_type', 'years_of_employment', 'company_type', 'house_type',
            'monthly_rent', 'family_size', 'dependents', 'school_fees', 'college_fees',
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
            'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
            'emergency_fund', 'emi_scenario', 'requested_amount', 'requested_tenure'
        ]
        input_df = input_df[all_features]

        # --- Load models only here ---
        rf_reg = joblib.load(r'C:\emi_prediction\Regression.pkl')       # Regression
        xgb_cls = joblib.load(r'C:\emi_prediction\Classification.pkl') # Classification

        # --- Predict ---
        predicted_emi = rf_reg.predict(input_df)[0]
        predicted_eligibility = xgb_cls.predict(input_df)[0]

        # Decode eligibility
        if 'emi_eligibility' in label_encoders:
            predicted_eligibility = label_encoders['emi_eligibility'].inverse_transform(
                [int(predicted_eligibility)]
            )[0]

        # Show results
        st.subheader("🧠 Prediction Results")
        st.metric("Predicted Max EMI Amount (₹)", f"{predicted_emi:,.2f}")
        st.metric("EMI Eligibility Status", predicted_eligibility)
        st.success("✅ Prediction Completed Successfully!")
