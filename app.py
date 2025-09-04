import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# --- Load Model and Scaler ---
# Use st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_model_and_scaler():
    """Loads the pre-trained XGBoost model and the scaler."""
    try:
        model = joblib.load('xgboost_concrete_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler files not found.")
        st.error("Please ensure 'xgboost_concrete_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model_and_scaler()

# --- App Title and Description ---
st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("""
This application predicts the compressive strength of concrete based on its components and age. 
Adjust the sliders and input values in the sidebar to match your concrete mixture, then click 'Predict Strength'.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Concrete Components")

def user_inputs():
    """Creates sidebar widgets and returns user inputs as a DataFrame."""
    cement = st.sidebar.slider('Cement (kg/m³)', 100.0, 550.0, 332.5)
    slag = st.sidebar.slider('Blast Furnace Slag (kg/m³)', 0.0, 360.0, 142.5)
    fly_ash = st.sidebar.slider('Fly Ash (kg/m³)', 0.0, 200.0, 0.0)
    water = st.sidebar.slider('Water (kg/m³)', 120.0, 250.0, 180.0)
    superplasticizer = st.sidebar.slider('Superplasticizer (kg/m³)', 0.0, 32.0, 6.0)
    coarse_agg = st.sidebar.slider('Coarse Aggregate (kg/m³)', 800.0, 1150.0, 970.0)
    fine_agg = st.sidebar.slider('Fine Aggregate (kg/m³)', 590.0, 1000.0, 770.0)
    age = st.sidebar.number_input('Age (days)', min_value=1, max_value=365, value=28)

    data = {
        'Cement': cement,
        'Blast Furnace Slag': slag,
        'Fly Ash': fly_ash,
        'Water': water,
        'Superplasticizer': superplasticizer,
        'Coarse Aggregate': coarse_agg,
        'Fine Aggregate': fine_agg,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_inputs()

# --- Main Panel: Display Inputs and Prediction ---
st.header("Your Input Values")
st.write(input_df)

if model and scaler:
    # Prediction button
    if st.button('Predict Strength', type="primary"):
        # Scale the user inputs using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make the prediction
        prediction = model.predict(input_scaled)
        predicted_strength = prediction[0]

        st.header("Prediction Result")
        st.success(f"**Predicted Concrete Strength: {predicted_strength:.2f} MPa**")

        # FIX: Ensure the progress value is a float between 0.0 and 1.0
        progress_value = min(float(predicted_strength) / 85.0, 1.0)

        # Add a gauge-style meter for visualization
        st.progress(progress_value, text=f"{predicted_strength:.2f} MPa")
        st.caption("The progress bar shows the predicted strength relative to a common maximum of 85 MPa.")
else:
    st.warning("Model is not loaded. Cannot make predictions.")

