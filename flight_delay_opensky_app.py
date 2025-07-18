import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix

# -------------------
# Load model and encoder
# -------------------
@st.cache_resource
def load_model():
    with open("delay_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("airline_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

# -------------------
# OpenSky API (No key needed)
# -------------------
def get_opensky_data(callsign):
    url = "https://opensky-network.org/api/states/all"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        for state in data.get("states", []):
            if callsign.upper() in str(state[1]).strip():
                return {
                    "callsign": state[1].strip(),
                    "origin_country": state[2],
                    "longitude": state[5],
                    "latitude": state[6],
                    "altitude": state[7],
                    "speed": state[9] * 3.6 if state[9] else None
                }
    except:
        return None
    return None

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Flight Delay Predictor + Live Tracker", layout="wide")
st.title("✈️ Flight Delay Predictor + 🛰️ Live Flight Tracker")

callsign = st.text_input("Enter Flight Callsign (e.g., AI202, UK752, IGO123):")

if st.button("📡 Track & Predict"):
    with st.spinner("Fetching live flight data..."):
        flight = get_opensky_data(callsign)

    if flight:
        st.success(f"Live Data for {flight['callsign']} from {flight['origin_country']}")
        st.map(pd.DataFrame({'lat': [flight['latitude']], 'lon': [flight['longitude']]}))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🛬 Altitude", f"{flight['altitude']:.0f} m" if flight['altitude'] else "N/A")
        with col2:
            st.metric("🚀 Speed", f"{flight['speed']:.0f} km/h" if flight['speed'] else "N/A")

        # For ML prediction - dummy values used for demo
        st.subheader("🔍 Predict Flight Delay")
        airline_code = st.selectbox("Airline Code", encoder.classes_.tolist())
        dep_delay = st.slider("Departure Delay (min)", 0, 180, 10)
        carrier = st.slider("Carrier Delay", 0, 100, 5)
        weather = st.slider("Weather Delay", 0, 100, 5)
        nas = st.slider("NAS Delay", 0, 100, 5)
        security = st.slider("Security Delay", 0, 50, 0)
        late_aircraft = st.slider("Late Aircraft Delay", 0, 200, 10)

        if st.button("✈️ Predict Delay"):
            try:
                airline_encoded = encoder.transform([airline_code])[0]
                features = np.array([[airline_encoded, dep_delay, carrier, weather, nas, security, late_aircraft]])
                prediction = model.predict(features)

                if prediction[0] == 1:
                    st.error("❌ Prediction: Flight is likely to be DELAYED")
                else:
                    st.success("✅ Prediction: Flight is likely to be ON TIME")

                # Graphs
                st.subheader("📊 Delay Breakdown")
                delay_vals = [carrier, weather, nas, security, late_aircraft]
                delay_labels = ["Carrier", "Weather", "NAS", "Security", "Late Aircraft"]
                st.plotly_chart(px.pie(names=delay_labels, values=delay_vals, hole=0.3))

                st.subheader("📈 Feature Importance")
                feat_names = ["AIRLINE", "DEP_DELAY", "CARRIER", "WEATHER", "NAS", "SECURITY", "LATE_AIRCRAFT"]
                importance = model.feature_importances_
                imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importance})
                fig_imp, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax)
                st.pyplot(fig_imp)

                st.subheader("🧪 Simulated Confusion Matrix")
                y_true = [0, 1, 1, 0, 1, 0, 1, 0]
                y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["On Time", "Delayed"],
                            yticklabels=["On Time", "Delayed"])
                st.pyplot(fig_cm)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Flight not found. Try a different callsign or wait a moment.")

st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Made with ❤️ by Harsh Mishra</h5>", unsafe_allow_html=True)