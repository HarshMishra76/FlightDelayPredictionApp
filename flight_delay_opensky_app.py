import streamlit as st
import pandas as pd
import requests
import pickle
import plotly.express as px
from geopy.distance import geodesic

# ----------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Flight Delay Dashboard", layout="wide", page_icon="🛫")

# ----------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    with open("delay_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("airline_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

# ----------------- SIDEBAR -------------------
st.sidebar.title("✈️ Flight Info")
callsign = st.sidebar.text_input("Enter Flight Callsign (e.g., AAL123)", "AAL123")
st.sidebar.info("🔍 Try callsigns like: AAL123, UAE203, AFR72B, BAW275")

st.title("📊 Flight Delay Prediction Dashboard")
st.markdown("Get live flight information and predict if your flight will be delayed.")

# ----------------- FETCH FLIGHT DATA -------------------
@st.cache_data(ttl=30)
def get_flight_data(callsign):
    url = "https://opensky-network.org/api/states/all"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        for state in data["states"]:
            if state[1] and callsign.strip().upper() in state[1]:
                return {
                    "callsign": state[1].strip(),
                    "origin_country": state[2],
                    "longitude": state[5],
                    "latitude": state[6],
                    "altitude": state[7],
                    "velocity": state[9],
                    "airline": state[1][:2],  # Extract first 2 characters as airline code
                }
    except Exception as e:
        st.error("🌐 Could not fetch flight data.")
        return None
    return None

flight_data = get_flight_data(callsign)

# ----------------- SHOW FLIGHT INFO -------------------
if flight_data:
    col1, col2, col3 = st.columns(3)
    col1.metric("✈️ Callsign", flight_data['callsign'])
    col2.metric("🌍 Country", flight_data['origin_country'])
    col3.metric("🚀 Speed (m/s)", round(flight_data['velocity'], 2) if flight_data['velocity'] else "N/A")

    st.metric("🛫 Altitude (m)", round(flight_data['altitude'], 2) if flight_data['altitude'] else "N/A")

    # ----------------- MAP -------------------
    if flight_data["latitude"] and flight_data["longitude"]:
        st.map(pd.DataFrame({
            "lat": [flight_data["latitude"]],
            "lon": [flight_data["longitude"]]
        }))
else:
    st.warning("🔍 Flight not found. Please try another callsign.")
    st.stop()

# ----------------- INPUT FOR PREDICTION -------------------
st.subheader("📥 Delay Prediction Inputs")

dep_delay = st.number_input("🕒 Departure Delay (in minutes)", min_value=0, value=5)
carrier_delay = st.number_input("🛫 Carrier Delay", min_value=0)
weather_delay = st.number_input("⛈️ Weather Delay", min_value=0)
nas_delay = st.number_input("📡 NAS Delay", min_value=0)
security_delay = st.number_input("🔒 Security Delay", min_value=0)
late_aircraft_delay = st.number_input("🛬 Late Aircraft Delay", min_value=0)

# ----------------- PREDICT -------------------
if st.button("🔮 Predict Flight Delay"):
    try:
        airline_encoded = encoder.transform([[flight_data['airline']]])[0]

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
            st.error("🔴 Your flight **might be delayed**.")
        else:
            st.success("🟢 Your flight is **on time**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------- CHARTS -------------------
st.subheader("📊 Delay Cause Breakdown")

chart_df = pd.DataFrame({
    "Type": ["Carrier", "Weather", "NAS", "Security", "Late Aircraft"],
    "Minutes": [carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay]
})

col1, col2 = st.columns(2)

with col1:
    st.write("📉 Bar Chart")
    fig_bar = px.bar(chart_df, x="Type", y="Minutes", color="Type", title="Delay Types")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.write("📈 Pie Chart")
    fig_pie = px.pie(chart_df, names="Type", values="Minutes", title="Delay Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

# ----------------- Footer -------------------
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Made with ❤️ by <b>Harsh Mishra</b></h5>", unsafe_allow_html=True)
