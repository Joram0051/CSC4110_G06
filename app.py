import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent  # app.py and the model files are in the repo root
MODEL_PATH = ROOT / "G06_model.joblib"
INFO_PATH = ROOT / "G06_model_info.json"

st.set_page_config(page_title="Air fare estimator", page_icon=":airplane:")
st.title("Safari Fare Desk: air fare estimator")
st.markdown("**Group G06** | CSC 4110 Machine Learning, Kibabii University")
st.write("Members: Daniel Natembeya, Ledama Nomali, Raynner Kavulika, Joram Sayia, Emmanuel Wekesa, Shikuku Cyrus")
st.write("Dataset: Flight Price Prediction (Clean_Dataset.csv, Kaggle; 300,153 Indian domestic flight fares)")


@st.cache_resource
def load_model():
    # load the saved pipeline and its information file once
    return joblib.load(MODEL_PATH), json.loads(INFO_PATH.read_text())


model, info = load_model()
ch = info["choices"]

col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", ch["airline"])
    source_city = st.selectbox("Source city", ch["source_city"])
    destination_city = st.selectbox("Destination city", ch["destination_city"], index=1)
    stops = st.selectbox("Stops", ch["stops"], index=1)
with col2:
    departure_time = st.selectbox("Departure time", ch["departure_time"], index=1)
    travel_class = st.selectbox("Class", ch["class"])
    days_left = st.slider("Days left before departure", 1, 49, 20)
    duration = st.number_input("Duration (hours)", min_value=0.5, max_value=50.0, value=10.0, step=0.25)

if st.button("Predict"):
    if source_city == destination_city:
        st.warning("Source and destination are the same city, so the estimate may be unreliable.")
    row = pd.DataFrame([{"airline": airline, "source_city": source_city, "destination_city": destination_city,
                         "stops": stops, "departure_time": departure_time, "class": travel_class,
                         "days_left": days_left, "duration": duration}])
    fare = float(model.predict(row)[0])
    rmse = info["test_rmse"]
    st.success(f"Estimated fare: {fare:,.0f} INR  (+/- {rmse:,.0f} INR)")
    st.caption(f"Likely range: {max(fare - rmse, 0):,.0f} to {fare + rmse:,.0f} INR. The range is the model's test RMSE.")

st.divider()
st.caption(f"Model: {info['model_name']}. Trained on {info['trained_on']} fare data. "
           "Fares change daily, so treat this as an estimate and confirm in the airline system.")
