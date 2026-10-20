import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Smart Energy Predictor", page_icon="⚡")

model = joblib.load("electricity_model.pkl")
scaler = joblib.load("electricity_scaler.pkl")

st.title("⚡ Smart Electricity Consumption Predictor")
st.write("Predict electricity consumption using an ANN model.")

appliances = st.number_input("Number of appliances", 1, 100, 5)
hours = st.number_input("Daily usage hours", 1, 24, 4)
previous_usage = st.number_input("Previous daily consumption (kWh)", min_value=0.1, value=6.0)

if st.button("Predict Consumption"):
    inputs = np.array([[appliances, hours, previous_usage]])
    scaled_inputs = scaler.transform(inputs)
    result = model.predict(scaled_inputs, verbose=0)[0][0]

    st.success(f"Predicted consumption: {result:.2f} kWh/day")
    st.caption("Estimate from a small practice dataset; not a real-world forecast.")
