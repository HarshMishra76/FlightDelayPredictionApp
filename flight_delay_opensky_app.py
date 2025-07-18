import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="centered")

# Title
st.title("✈️ Flight Delay Predictor")
st.markdown("Use this app to predict whether a flight will be delayed based on various delay causes.")

# Load model and encoder
@st.cache_resource
def load_model():
    with open("delay_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("airline_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

# UI Inputs
airline_classes = encoder.classes_.tolist()
airline = st.selectbox("Airline Code", airline_classes)

dep_delay = st.number_input("Departure Delay (in minutes)", min_value=0)
carrier_delay = st.number_input("Carrier Delay", min_value=0)
weather_delay = st.number_input("Weather Delay", min_value=0)
nas_delay = st.number_input("NAS Delay", min_value=0)
security_delay = st.number_input("Security Delay", min_value=0)
late_aircraft_delay = st.number_input("Late Aircraft Delay", min_value=0)

# ------------------- Prediction ------------------- #
if st.button("🧾 Predict Delay"):
    try:
        # Encode airline
        airline_encoded = encoder.transform([[airline]])[0]

        # Input DataFrame (must match model training columns exactly)
        input_data = pd.DataFrame([[
            airline_encoded,
            dep_delay,
            carrier_delay,
            weather_delay,
            nas_delay,
            security_delay,
            late_aircraft_delay
        ]], columns=[
            "AIRLINE",
            "DEP_DELAY",
            "DELAY_DUE_CARRIER",
            "DELAY_DUE_WEATHER",
            "DELAY_DUE_NAS",
            "DELAY_DUE_SECURITY",
            "DELAY_DUE_LATE_AIRCRAFT"
        ])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🔴 Prediction: Your flight **might be delayed**.")
        else:
            st.success("🟢 Prediction: Your flight is **on time**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by <b>Harsh Mishra</b></p>", unsafe_allow_html=True)
