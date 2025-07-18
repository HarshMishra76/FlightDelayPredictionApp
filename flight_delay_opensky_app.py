import streamlit as st
import pandas as pd
import requests
import pickle
import plotly.express as px
from geopy.distance import geodesic

# ------------------- CONFIG ---------------------
st.set_page_config(page_title="Flight Delay Dashboard", layout="wide", page_icon="🛫")

# ------------------- MODEL LOAD ---------------------
@st.cache_resource
def load_model():
    with open("delay_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("airline_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

# ------------------- SIDEBAR WITH DROPDOWN ---------------------
st.sidebar.title("✈️ Flight Input")

# Trained airlines from model encoder
trained_airlines = list(encoder.classes_)

# Sample mapped callsigns per airline
airline_options = {
    "AA (American Airlines)": "AAL123",
    "UA (United Airlines)": "UAL456",
    "DL (Delta Airlines)": "DAL789",
    "WN (Southwest Airlines)": "SWA123",
    "B6 (JetBlue Airways)": "JBU456",
    "AS (Alaska Airlines)": "ASA789"
}

# Filter only trained ones
filtered_options = {k: v for k, v in airline_options.items() if k.split(" ")[0] in trained_airlines}

selected_label = st.sidebar.selectbox("✈️ Choose a Flight (Dropdown)", list(filtered_options.keys()))
callsign = filtered_options.get(selected_label, "AAL123")

# ------------------- FLIGHT FETCH FUNCTION ---------------------
@st.cache_data(ttl=30)
def get_flight_data(callsign):
    url = "https://opensky-network.org/api/states/all"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        for state in data["states"]:
            if state[1] and callsign.strip().upper() == state[1].strip():
                return {
                    "callsign": state[1].strip(),
                    "origin_country": state[2],
                    "longitude": state[5],
                    "latitude": state[6],
                    "altitude": state[7],
                    "velocity": state[9],
                    "airline": state[1][:2] if state[1][:2] in encoder.classes_ else None
                }
    except Exception as e:
        st.error("🌐 Could not fetch flight data.")
        return None
    return None

# ------------------- FETCH + VALIDATE ---------------------
flight_data = get_flight_data(callsign)

if not flight_data or not flight_data["airline"]:
    st.warning("⚠️ This airline is not supported in the prediction model.")
    st.stop()

# ------------------- TITLE ---------------------
st.title("🛫 Flight Delay Prediction Dashboard")
st.markdown("Track real-time flight info and predict delays using machine learning.")

# ------------------- FLIGHT METRICS ---------------------
col1, col2, col3 = st.columns(3)
col1.metric("✈️ Callsign", flight_data['callsign'])
col2.metric("🌍 Country", flight_data['origin_country'])
col3.metric("🚀 Speed (m/s)", round(flight_data['velocity'], 2) if flight_data['velocity'] else "N/A")
st.metric("🛫 Altitude (m)", round(flight_data['altitude'], 2) if flight_data['altitude'] else "N/A")

# ------------------- MAP ---------------------
if flight_data["latitude"] and flight_data["longitude"]:
    st.map(pd.DataFrame({
        "lat": [flight_data["latitude"]],
        "lon": [flight_data["longitude"]]
    }))

# ------------------- INPUTS ---------------------
st.subheader("📥 Delay Input Parameters")

dep_delay = st.number_input("Departure Delay (minutes)", min_value=0, value=5)
carrier_delay = st.number_input("Carrier Delay", min_value=0)
weather_delay = st.number_input("Weather Delay", min_value=0)
nas_delay = st.number_input("NAS Delay", min_value=0)
security_delay = st.number_input("Security Delay", min_value=0)
late_aircraft_delay = st.number_input("Late Aircraft Delay", min_value=0)

# ------------------- PREDICT ---------------------
if st.button("🔮 Predict Flight Delay"):
    try:
        airline_encoded = encoder.transform([[flight_data["airline"]]])[0]

        input_df = pd.DataFrame([[
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

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("🔴 Prediction: Your flight **might be delayed**.")
        else:
            st.success("🟢 Prediction: Your flight is **on time**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------- GRAPHS ---------------------
st.subheader("📊 Delay Components Breakdown")

chart_df = pd.DataFrame({
    "Type": ["Carrier", "Weather", "NAS", "Security", "Late Aircraft"],
    "Minutes": [carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay]
})

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(px.bar(chart_df, x="Type", y="Minutes", title="Delay Reasons", color="Type"), use_container_width=True)

with col2:
    st.plotly_chart(px.pie(chart_df, names="Type", values="Minutes", title="Delay Distribution"), use_container_width=True)

# ------------------- FOOTER ---------------------
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Made with ❤️ by <b>Harsh Mishra</b></h5>", unsafe_allow_html=True)
