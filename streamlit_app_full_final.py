import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from geopy.distance import geodesic

# Load model and encoder
@st.cache_resource
def load_model():
    with open("delay_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    with open("airline_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

st.set_page_config(page_title="Flight Delay Predictor", layout="centered")
st.title("✈️ Flight Delay Prediction App")
st.markdown("Enter flight delay details below to predict if it will be delayed:")

# Airline full names
airline_full_names = {
    'AA': 'American Airlines',
    'DL': 'Delta Airlines',
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines'
}

# Inputs
valid_airlines = encoder.classes_
airline = st.selectbox("Airline Code", valid_airlines)
selected_name = airline_full_names.get(airline, "Unknown Airline")
st.info(f"🛫 You selected: **{selected_name} ({airline})**")

dep_delay = st.slider("Departure Delay (mins)", 0, 180, 10)
carrier = st.slider("Carrier Delay (mins)", 0, 100, 5)
weather = st.slider("Weather Delay (mins)", 0, 100, 7)
nas = st.slider("NAS Delay (mins)", 0, 100, 4)
security = st.slider("Security Delay (mins)", 0, 50, 3)
late_aircraft = st.slider("Late Aircraft Delay (mins)", 0, 200, 5)

# Delay severity meter
total_delay = dep_delay + carrier + weather + nas + security + late_aircraft
severity = "🟢 Low" if total_delay < 15 else "🟡 Medium" if total_delay < 40 else "🔴 High"
st.metric(label="Estimated Delay Severity", value=severity)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Predict Delay"):
    try:
        airline_encoded = encoder.transform([airline])[0]
        input_data = np.array([[airline_encoded, dep_delay, carrier, weather, nas, security, late_aircraft]])
        result = model.predict(input_data)

        if result[0] == 1:
            st.error("❌ Flight is likely to be DELAYED.")
            st.snow()
        else:
            st.success("✅ Flight is likely to be ON TIME.")
            st.balloons()

        st.session_state.history.append((airline, dep_delay, result[0]))

        # Pie chart
        st.subheader("📊 Delay Contribution Breakdown")
        delay_values = [carrier, weather, nas, security, late_aircraft]
        delay_labels = ["Carrier", "Weather", "NAS", "Security", "Late Aircraft"]
        fig_pie = px.pie(names=delay_labels, values=delay_values, title="Delay Causes", hole=0.4)
        st.plotly_chart(fig_pie)

        # Feature importance
        st.subheader("📈 Feature Importance (Model Insight)")
        feature_names = ["AIRLINE", "DEP_DELAY", "CARRIER", "WEATHER", "NAS", "SECURITY", "LATE_AIRCRAFT"]
        importance = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        fig_imp, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax)
        ax.set_title("Features impacting prediction")
        st.pyplot(fig_imp)

        # Confusion Matrix
        st.subheader("🧪 Confusion Matrix (Sample)")
        y_true = [0, 1, 1, 0, 1, 1, 0, 0]
        y_pred = [0, 1, 1, 0, 0, 1, 0, 1]
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["On Time", "Delayed"], yticklabels=["On Time", "Delayed"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Dynamic prediction visual
        st.subheader("🎯 Prediction Summary Visual")
        if result[0] == 1:
            fig = px.bar(x=delay_labels, y=delay_values, title="Contributors to Delay")
        else:
            fig = px.pie(names=["On Time", "Buffer Time"], values=[1, 2], title="Confidence Chart", hole=0.5)
        st.plotly_chart(fig)

    except Exception as e:
        st.error("⚠️ Prediction Error: Airline not recognized by the model.")

# History
with st.expander("📜 Prediction History"):
    for entry in st.session_state.history:
        st.write(f"✈️ Airline: {entry[0]}, DepDelay: {entry[1]} → {'Delayed' if entry[2]==1 else 'On Time'}")

# Flight Route Map
st.subheader("🗺️ Visualize Flight Route on Map")
airports = {
    "Delhi (DEL)": (28.5562, 77.1000),
    "Mumbai (BOM)": (19.0896, 72.8656),
    "Kolkata (CCU)": (22.6547, 88.4467),
    "Bangalore (BLR)": (13.1986, 77.7066),
    "Hyderabad (HYD)": (17.2403, 78.4294),
    "Chennai (MAA)": (12.9941, 80.1709),
    "Ahmedabad (AMD)": (23.0732, 72.6347)
}

from_city = st.selectbox("From Airport", list(airports.keys()), index=0)
to_city = st.selectbox("To Airport", list(airports.keys()), index=1)

if from_city == to_city:
    st.warning("Please select two different airports.")
else:
    src_lat, src_lon = airports[from_city]
    dst_lat, dst_lon = airports[to_city]
    distance = geodesic((src_lat, src_lon), (dst_lat, dst_lon)).km
    st.markdown(f"**Distance:** `{distance:.2f} km` between {from_city} and {to_city}")

    fig = go.Figure(go.Scattergeo(
        lon = [src_lon, dst_lon],
        lat = [src_lat, dst_lat],
        mode = 'lines+markers+text',
        line = dict(width = 2, color = 'blue'),
        marker = dict(size = 8, color='red'),
        text = [from_city, to_city],
        textposition = "top center"
    ))

    fig.update_layout(
        geo = dict(
            scope='asia',
            projection_type='equirectangular',
            showland = True,
            landcolor = 'lightgray',
        ),
        title = f"Route Map: {from_city} ✈️ {to_city}",
        margin={"r":0,"t":30,"l":0,"b":0}
    )

    st.plotly_chart(fig, use_container_width=True)