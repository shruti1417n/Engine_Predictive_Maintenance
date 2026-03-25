import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
try:
    model_path = hf_hub_download(repo_id="ShrutiHulyal/Engine-Predictive-Maintenance-model", filename="best_engine_model_v1.joblib")
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure the model exists in the Hugging Face repo.")
    st.stop()

# Streamlit UI for Engine Predictive Maintenance
st.title("Engine Predictive Maintenance App")
st.write("""
This application predicts whether an engine requires maintenance (Faulty) or is operating normally.
Input the engine's current sensor readings below.
""")

# User input fields
st.header("Engine Sensor Readings")
Engine_rpm = st.number_input("Engine RPM", min_value=0.0, max_value=3000.0, value=750.0, step=10.0)
Lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
Fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, max_value=30.0, value=6.0, step=0.1)
Coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
lub_oil_temp = st.number_input("Lubricating Oil Temperature (°C)", min_value=0.0, max_value=150.0, value=75.0, step=1.0)
Coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)

# Assemble input into DataFrame, ensuring correct column order and names as per Xtrain
input_data = pd.DataFrame([{
    'Engine_rpm': Engine_rpm,
    'Lub_oil_pressure': Lub_oil_pressure,
    'Fuel_pressure': Fuel_pressure,
    'Coolant_pressure': Coolant_pressure,
    'lub_oil_temp': lub_oil_temp,
    'Coolant_temp': Coolant_temp
}])


if st.button("Predict Engine Condition"):
    # Predict probability
    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Use the classification threshold from training (0.45)
    classification_threshold = 0.45 # This threshold should ideally be determined during model training
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]

    result = "Faulty (Requires Maintenance)" if prediction == 1 else "Normal (Operating Correctly)"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"The model predicts: **{result}** (Probability of Faulty: {prediction_proba[0]:.2f})")
    else:
        st.success(f"The model predicts: **{result}** (Probability of Faulty: {prediction_proba[0]:.2f})")
