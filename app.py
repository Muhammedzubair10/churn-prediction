import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
# Load the column names from training
model_columns = joblib.load('model_columns.pkl') # Assumed to be saved during model training

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Create input fields for user
# This is a simplified version. A real app would need inputs for all 68 features.
# For simplicity, we'll show a few key inputs.
st.sidebar.header("Customer Input Features")
international_plan = st.sidebar.selectbox("International Plan?", ("No", "Yes"))
customer_service_calls = st.sidebar.slider("Number of Customer Service Calls", 0, 10, 2)
total_day_minutes = st.sidebar.slider("Total Day Minutes", 0.0, 400.0, 180.0)
total_eve_minutes = st.sidebar.slider("Total Evening Minutes", 0.0, 400.0, 200.0)

# --- Create a dataframe from inputs ---
# This is complex because of one-hot encoding. A real implementation would be more robust.
# For this simulation, we'll create a placeholder dataframe.
input_data = pd.DataFrame(columns=model_columns)
input_data.loc[0] = 0 # Initialize with zeros
input_data['international_plan'] = 1 if international_plan == "Yes" else 0
input_data['customer_service_calls'] = customer_service_calls
input_data['total_day_minutes'] = total_day_minutes
input_data['total_eve_minutes'] = total_eve_minutes

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Prediction")
churn_status = "Yes" if prediction[0] == 1 else "No"
st.write(f"The customer is likely to churn: **{churn_status}**")

st.subheader("Prediction Probability")
st.write(f"Probability of Not Churning: **{prediction_proba[0][0]:.2f}**")
st.write(f"Probability of Churning: **{prediction_proba[0][1]:.2f}**")
