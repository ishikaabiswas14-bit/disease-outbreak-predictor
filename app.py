import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Outbreak Severity Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #1e293b 100%);
    color: white;
}
header {
    visibility: hidden;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}
.gradient-text {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 35px;
}
.card {
    background: rgba(255,255,255,0.07);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}
.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 15px;
}
.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-size: 18px;
    font-weight: 600;
}
.success-box {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
.danger-box {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #38bdf8;
}
.metric-label {
    color: #cbd5e1;
    margin-top: 5px;
}
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 13px;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-title">
🦟 <span class="gradient-text">Disease Outbreak Severity Predictor</span>
</div>
<div class="hero-subtitle">
AI-Powered Early Warning System for Indian Health Authorities
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = joblib.load('severity_model_final.pkl')
    le_disease = joblib.load('le_disease.pkl')
    le_state = joblib.load('le_state.pkl')
    le_district = joblib.load('le_district.pkl')
    return model, le_disease, le_state, le_district

try:
    model, le_disease, le_state, le_district = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

with st.sidebar:
    st.markdown("### ℹ️ About This Tool")
    st.markdown("Predicts outbreak severity based on disease, location, climate, and death reports.")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("- **Model:** Random Forest Classifier")
    st.markdown("- **Accuracy:** 78.1%")
    st.markdown("- **Data Source:** EpiClim (2009-2022)")
    st.markdown("- **SDG Alignment:** SDG 3 - Target 3.d")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Disease Information")
    disease = st.selectbox("Select Disease", le_disease.classes_)
    state = st.selectbox("Select State", le_state.classes_)
    district = st.selectbox("Select District", le_district.classes_)

with col2:
    st.markdown("### 🌦️ Environmental Factors")
    month = st.slider("Month", 1, 12, 6)
    week = st.slider("Week", 1, 52, 26)
    temperature = st.number_input("Temperature (Kelvin)", value=300.0)
    precipitation = st.number_input("Precipitation (mm)", value=50.0)
    latitude = st.number_input("Latitude", value=20.0)
    longitude = st.number_input("Longitude", value=78.0)
    had_deaths = st.radio("Deaths Reported?", ["No", "Yes"], horizontal=True)

if st.button("🔮 Predict Severity", use_container_width=True):
    with st.spinner("Analyzing..."):
        time.sleep(1)
        disease_code = le_disease.transform([disease])[0]
        state_code = le_state.transform([state])[0]
        district_code = le_district.transform([district])[0]
        deaths_binary = 1 if had_deaths == "Yes" else 0
        
        input_data = [[disease_code, state_code, district_code, month, week, 
                       temperature, precipitation, latitude, longitude, deaths_binary]]
        
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        risk = proba[1] * 100
    
    if prediction == 1:
        st.markdown(f'<div class="danger-box">🚨 SEVERE OUTBREAK<br>Risk: {risk:.1f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-box">✅ NON-SEVERE<br>Risk: {risk:.1f}%</div>', unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{proba[prediction]*100:.1f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{"HIGH" if prediction==1 else "LOW"}</div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{risk:.1f}%</div><div class="metric-label">Severity Score</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Disease Outbreak Severity Predictor • Random Forest • EpiClim (2009-2022) • SDG 3 - Target 3.d</div>', unsafe_allow_html=True)
