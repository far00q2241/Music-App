import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("music_model.pkl")  # change filename if needed

st.set_page_config(page_title="Music Listening Predictor", layout="centered")

st.title("🎧 Daily Music Listening Time Predictor")
st.write("Predict how many minutes a user listens to music daily based on behavior.")

# Input fields
age = st.number_input("Age", min_value=10, max_value=80, value=25)

songs_per_day = st.number_input("Songs per Day", min_value=0, value=50)

playlists_count = st.number_input("Playlists Count", min_value=0, value=10)

skip_rate_pct = st.slider("Skip Rate (%)", 0, 100, 30)

# Predict button
if st.button("Predict Listening Time"):

    input_data = np.array([[age, songs_per_day, playlists_count, skip_rate_pct]])

    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Daily Listening Minutes: {prediction:.2f} minutes")

    st.write("---")
    st.write("### Input Summary")
    st.write(pd.DataFrame(input_data, columns=[
        "age", "songs_per_day", "playlists_count", "skip_rate_pct"
    ]))
