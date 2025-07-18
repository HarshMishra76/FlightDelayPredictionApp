import streamlit as st
import pandas as pd
import requests
import pickle

# ------------------- Load Model & Encoder ------------------- #
@st.cache_resource
def load_model():
    with open("delay_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("airline_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

# ------------------- App Title ------------------- #
st.set_page_config(page_title="Flight Delay Predictor", layout="wide")
st.title("✈️ Real-Time Flight Delay Prediction App")
st.markdown("Powered by **OpenSky API** and Machine Learning 🚀")

# ------------------- Fetch Live Flights ------------------- #
st.header("📡 Live Flight Tracker")
try:
    response = requests.get("https://opensky-network.org/api/states/all", timeout=10)
    data = response.json()
    flights = data.get("states", [])
    callsigns = sorted(list(set([f[1].strip() for f in flights if f[1] and f[1].strip()])))
    
    selected_callsign = st.selectbox("Select a live flight (callsign)", callsigns)
    selected_flight = next((f for f in flights if f[1].strip() == selected_callsign), None)

    if selected_flight:
        st.success(f"✈️ {selected_callsign} is currently flying.")
        st.write(f"**Altitude:** {selected_flight[7]} meters")
        st.write(f"**Ground Speed:** {selected_flight[9]} m/s")
    else:
        st.warning("Flight not found. Try again in a moment.")
except Exception as e:
    st.error(f"Unable to fetch flights: {e}")

# ------------------- Prediction Inputs ------------------- #
st.header("🧠 Predict Flight Delay")

airline_classes = encoder.classes_.tolist()
airline = st.selectbox("Airline Code", airline_classes)
day_of_month = st.slider("Day of Month", 1, 31, 15)
departure_time = st.slider("Scheduled Departure Time (hhmm)", 0, 2359, 900)
carrier_delay = st.number_input("Carrier Delay (mins)", 0, 300, 0)
weather_delay = st.number_input("Weather Delay (mins)", 0, 300, 0)
nas_delay = st.number_input("NAS Delay (mins)", 0, 300, 0)
security_delay = st.number_input("Security Delay (mins)", 0, 300, 0)
late_aircraft_delay = st.number_input("Late Aircraft Delay (mins)", 0, 300, 0)

# ------------------- Prediction ------------------- #
if st.button("🧾 Predict Delay"):
    try:
if st.button("🧾 Predict Delay"):
    try:
        input_data = pd.DataFrame([[
            day_of_month,
            departure_time,
            carrier_delay,
            weather_delay,
            nas_delay,
            security_delay,
            late_aircraft_delay
        ]], columns=[
            "DAY_OF_MONTH", "DEP_TIME", "CARRIER_DELAY", "WEATHER_DELAY",
            "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
        ])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🔴 Prediction: Your flight **might be delayed**.")
        else:
            st.success("🟢 Prediction: Your flight is **on time**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🔴 Prediction: Your flight **might be delayed**.")
        else:
            st.success("🟢 Prediction: Your flight is **on time**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------- Footer ------------------- #
st.markdown("---")
st.markdown("Made with ❤️ by **Harsh Mishra**")
