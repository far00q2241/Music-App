import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Music Listening Predictor",
    page_icon="🎧",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("music_model.pkl")

model = load_model()

st.title("🎧 Daily Music Listening Time Predictor")
st.write("Predict how many minutes a user listens to music daily based on user behavior.")

age = st.number_input(
    "Age",
    min_value=10,
    max_value=80,
    value=25
)

songs_per_day = st.number_input(
    "Songs Per Day",
    min_value=0,
    max_value=200,
    value=50
)

playlists_count = st.number_input(
    "Playlists Count",
    min_value=0,
    max_value=100,
    value=10
)

skip_rate_pct = st.slider(
    "Skip Rate (%)",
    min_value=0,
    max_value=100,
    value=30
)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[age, songs_per_day, playlists_count, skip_rate_pct]],
        columns=[
            "age",
            "songs_per_day",
            "playlists_count",
            "skip_rate_pct"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Daily Listening Time: {prediction:.2f} minutes")

    st.subheader("Input Summary")
    st.dataframe(input_data)
