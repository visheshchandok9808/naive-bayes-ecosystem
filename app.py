import streamlit as st
import pickle
import numpy as np

# Load the trained Naive Bayes model
with open("/Users/vishesh/Downloads/GreenAi/Naive bayes/naive_bayes_ecosystem_model.pkl", "rb") as file:
    model = pickle.load(file)

# Mapping output labels
label_map = {0: 'Healthy ✅', 1: 'At Risk ⚠️', 2: 'Degraded ❌'}

# Streamlit page config
st.set_page_config(page_title="Ecosystem Health Predictor", layout="centered")
st.title("🌿 Ecosystem Health Prediction App")
st.markdown("Enter the environmental data to predict the **ecosystem health** status.")

# Input form
with st.form("input_form"):
    st.subheader("🧪 Enter Input Features:")
    col1, col2 = st.columns(2)

    with col1:
        water_quality = st.number_input("Water Quality", min_value=0.0, max_value=100.0, value=50.0)
        biodiversity_index = st.number_input("Biodiversity Index (0-1)", min_value=0.0, max_value=1.0, value=0.5)
        soil_ph = st.number_input("Soil pH (0-14)", min_value=0.0, max_value=14.0, value=6.5)

    with col2:
        air_quality_index = st.number_input("Air Quality Index", min_value=0.0, max_value=500.0, value=100.0)
        vegetation_cover = st.number_input("Vegetation Cover (%)", min_value=0.0, max_value=100.0, value=50.0)

    submitted = st.form_submit_button("🔍 Predict Ecosystem Health")

    if submitted:
        input_features = np.array([[water_quality, air_quality_index, biodiversity_index, vegetation_cover, soil_ph]])
        prediction = model.predict(input_features)[0]
        st.success(f"🧾 Predicted Ecosystem Health: **{label_map[prediction]}**")
