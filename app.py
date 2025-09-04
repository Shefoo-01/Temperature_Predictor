import streamlit as st
import numpy as np
import joblib
import datetime

# Load saved objects
pipline=joblib.load('C:\\Users\\hp\\PycharmProjects\\PythonProject\\pipeline_temp.pkl')




# ---- Prediction Function ----
def predict_temp(hum, wind_S, surface_P, date, hour):
    # Extract features
    day_of_year = date.timetuple().tm_yday
    month = date.month

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Features array (2D)
    x = [[hum, wind_S, surface_P, hour_sin,hour_cos, day_sin,day_cos, month_sin,month_cos]]

    y_pred = pipline.predict(x)
    return float(y_pred[0])

# ---- Custom CSS (Dark + Glossy Red/Black Button) ----
st.markdown("""
<style>
:root { --accent-red: #ff2d55; --deep-red: #b3001b; --blk: #0b0b0c; }
html, body, [class^="css"]  { background-color: #101214; }
.main { background-color: #101214; color: #e9e9e9; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

h1, h2, h3 { color: #f5f5f7; }
.small { color:#a8a8a8; font-size:0.9rem; }

.card {
  background: #15171a;
  border: 1px solid #23252b;
  padding: 1.2rem;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.stSlider label, .stDateInput label { color:#d6d6d6 !important; }

/* Glossy red-on-black button */
.stButton>button {
  position: relative;
  background: linear-gradient(180deg, #131315 0%, #050506 100%);
  color: #ffffff;
  border: 1px solid rgba(255,45,85,0.45);
  border-radius: 14px;
  padding: 0.85rem 1.25rem;
  font-weight: 800;
  letter-spacing: .3px;
  text-transform: none;
  box-shadow: 0 8px 22px rgba(255,45,85,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
  transition: all .18s ease;
}
.stButton>button:before {
  content:'';
  position:absolute; inset:0 0 auto 0; height:50%;
  background: linear-gradient(180deg, rgba(255,255,255,0.22), rgba(255,255,255,0));
  border-top-left-radius:14px; border-top-right-radius:14px;
  pointer-events:none;
}
.stButton>button:hover {
  transform: translateY(-1px);
  border-color: var(--accent-red);
  box-shadow: 0 14px 32px rgba(255,45,85,0.55), inset 0 1px 0 rgba(255,255,255,0.05);
}
.stButton>button:active {
  transform: translateY(0);
  box-shadow: 0 10px 24px rgba(255,45,85,0.45);
}

/* Tag pills */
.tag {
  display:inline-block; background:#1e2025; border:1px solid #2b2d33;
  border-radius:999px; padding:.25rem .6rem; font-size:.8rem; color:#cfcfcf;
}
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown("<h1>🌡 Temperature Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='small'>Enter conditions, pick a date & hour, then get your model's temperature prediction.</p>", unsafe_allow_html=True)

# ---- Inputs (Sliders instead of number_input) ----
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        hum = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        st.markdown(f"<span class='tag'>Current: {hum:.1f}%</span>", unsafe_allow_html=True)

    with c2:
        wind_S = st.slider("Wind Speed (km/hr)", min_value=0.0, max_value=40.0, value=5.0, step=0.1)
        st.markdown(f"<span class='tag'>Current: {wind_S:.1f} m/s</span>", unsafe_allow_html=True)

    with c3:
        surface_P = st.slider("Surface Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0, step=0.1)
        st.markdown(f"<span class='tag'>Current: {surface_P:.1f} hPa</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Date & Hour
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c4, c5 = st.columns([2,1])
    with c4:
        date = st.date_input("Select Date", datetime.date.today())
    with c5:
        hour = st.slider("Hour of Day", 0, 23, 12)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Predict Button ----
if st.button("Predict Temperature"):
    temp = predict_temp(float(hum), float(wind_S), float(surface_P), date, int(hour))
    st.success(f"Predicted Temperature: {temp:.2f} °C")
