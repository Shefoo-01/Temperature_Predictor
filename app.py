import streamlit as st
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="🌡️ Temperature Predictor", page_icon="🔥", layout="wide")

# ---------------------------
# CSS Styling
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #ffffff 0%, #87CEEB 100%); /* White to sky blue gradient */
    }
    .card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(17, 24, 39, 0.08);
        border: 1px solid rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .title {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .subtitle {
        color: #475569;
        margin-top: -6px;
        margin-bottom: 16px;
    }
    label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    /* Subheader color in cards */
    .card .css-1d391kg h2 {
        color: black ;
    }
    /* Prediction Button Styling */
    div.stButton > button {
        background-color: black;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: black; /* Keep button black */
        border-bottom: 4px solid #1d4ed8; /* blue underline glow */
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.6); /* blue glow shadow */
    }
    /* Result Text Black */
    .result-text {
        color: black;
        font-size: 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("pipeline_temp.pkl")

pipeline = load_model()

# ---------------------------
# Prediction Function
# ---------------------------
def predict_temp(hum, wind_S, surface_P, date, hour):
    day_of_year = date.timetuple().tm_yday
    month = date.month

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    x = [[hum, wind_S, surface_P, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]]
    y_pred = pipeline.predict(x)
    return float(y_pred[0])

# ---------------------------
# Title
# ---------------------------
st.markdown('<div class="title"><span style="color:initial">🌡️</span> Temperature Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the expected temperature based on weather conditions</div>', unsafe_allow_html=True)

# ---------------------------
# Card 1: Date & Hour
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📅 Date & Time")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Date", value=datetime.date.today())
with col2:
    hour = st.slider("Hour of the Day", 0, 23, datetime.datetime.now().hour)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Card 2: Weather Conditions
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🌬️ Weather Conditions")

col3, col4, col5 = st.columns(3)
with col3:
    hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
with col4:
    wind_S = st.number_input("Wind Speed (km/h)", min_value=0.0, value=3.0, step=0.1)
with col5:
    surface_P = st.number_input("Surface Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0, step=0.1)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Card 3: Prediction
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Prediction")

if st.button("Predict Temperature"):
    try:
        result = predict_temp(hum, wind_S, surface_P, date, hour)
        st.markdown(f'<p class="result-text"><span style="color:initial">🌡️</span> Expected Temperature: {result:.2f} °C</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
