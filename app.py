import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('churn_model.joblib')
scaler = joblib.load('scaler.joblib')
train_cols = joblib.load('train_columns.joblib')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("👨‍💼 Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to churn based on their details. Please enter the customer's information below.")

try:
    df_original = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_original = df_original.drop('customerID', axis=1)
except FileNotFoundError:
    st.error("Original CSV file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Customer Details")
    gender = st.selectbox("Gender", df_original['gender'].unique())
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], help="0 for No, 1 for Yes")
    partner = st.selectbox("Partner", df_original['Partner'].unique())
    dependents = st.selectbox("Dependents", df_original['Dependents'].unique())
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.header("Service Details")
    phone_service = st.selectbox("Phone Service", df_original['PhoneService'].unique())
    multiple_lines = st.selectbox("Multiple Lines", df_original['MultipleLines'].unique())
    internet_service = st.selectbox("Internet Service", df_original['InternetService'].unique())
    online_security = st.selectbox("Online Security", df_original['OnlineSecurity'].unique())
    online_backup = st.selectbox("Online Backup", df_original['OnlineBackup'].unique())
    device_protection = st.selectbox("Device Protection", df_original['DeviceProtection'].unique())
    tech_support = st.selectbox("Tech Support", df_original['TechSupport'].unique())

with col3:
    st.header("Payment Details")
    streaming_tv = st.selectbox("Streaming TV", df_original['StreamingTV'].unique())
    streaming_movies = st.selectbox("Streaming Movies", df_original['StreamingMovies'].unique())
    contract = st.selectbox("Contract", df_original['Contract'].unique())
    paperless_billing = st.selectbox("Paperless Billing", df_original['PaperlessBilling'].unique())
    payment_method = st.selectbox("Payment Method", df_original['PaymentMethod'].unique())
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, format="%.2f")
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0, format="%.2f")

if st.button("Predict Churn", type="primary"):
    user_data = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([user_data])
    
    input_df['TotalCharges_log'] = np.log1p(input_df['TotalCharges'])

    input_df_encoded = pd.get_dummies(input_df)
    
    input_aligned = input_df_encoded.reindex(columns=train_cols, fill_value=0)
    
    input_scaled = scaler.transform(input_aligned)
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"**Customer will Churn** (Probability: {prediction_proba[0][1]:.2%})")
        st.write("Recommendation: Offer a discount or a loyalty bonus to retain this customer.")
    else:
        st.success(f"**Customer will NOT Churn** (Probability: {prediction_proba[0][0]:.2%})")