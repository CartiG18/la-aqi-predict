import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Config
st.set_page_config(page_title="LA AQI Forecaster", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_la_aqi.csv"
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "aqi_model.joblib"

# Load Resources
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# Header
st.title("Los Angeles AQI Forecasting System")
st.markdown("""
This dashboard visualizes real-time Air Quality Index (AQI) data from PurpleAir sensors 
and provides a machine learning interface to predict PM2.5 levels based on environmental factors.
""")

if df is None:
    st.error("Data not found. Please run the ingestion pipeline.")
    st.stop()

# --- ROW 1: METRICS & MAP ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Current Snapshot")
    avg_pm25 = df['pm2_5'].mean()
    avg_temp = df['temperature'].mean()
    st.metric("Avg PM2.5 (µg/m³)", f"{avg_pm25:.2f}")
    st.metric("Avg Temperature (°F)", f"{avg_temp:.1f}")
    st.metric("Active Sensors", len(df))

with col2:
    st.subheader("Sensor Map (PM2.5 Levels)")
    
    # Color scale for map (Green to Red)
    # PM2.5 > 35 is Unhealthy (Orange/Red)
    # We visualize this using pydeck
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitude", "latitude"],
        get_color="[255, (1 - pm2_5 / 50) * 255, 0, 160]", # Simple gradient logic
        get_radius=200,
        pickable=True,
    )
    
    view_state = pdk.ViewState(
        latitude=34.05,
        longitude=-118.25,
        zoom=9,
        pitch=0,
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "PM2.5: {pm2_5}\nTemp: {temperature}°F"}
    ))

# --- ROW 2: PREDICTION INTERFACE ---
st.divider()
st.subheader("🔮 Model Prediction Lab")
st.markdown("Adjust the sliders to simulate environmental conditions and predict pollution levels.")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    input_temp = st.slider("Temperature (°F)", 40.0, 110.0, 75.0)
    input_humidity = st.slider("Humidity (%)", 0.0, 100.0, 45.0)

with pred_col2:
    if model:
        # Feature Engineering (must match training)
        features = pd.DataFrame([{
            'temperature': input_temp,
            'humidity': input_humidity,
            'temp_humidity_interaction': input_temp * input_humidity
        }])
        
        prediction = model.predict(features)[0]
        
        # Color coding result
        if prediction <= 12: color = "green"
        elif prediction <= 35: color = "orange"
        else: color = "red"
        
        st.markdown(f"### Predicted PM2.5: :{color}[{prediction:.2f} µg/m³]")
        
        if prediction <= 12: st.success("Air Quality: GOOD")
        elif prediction <= 35: st.warning("Air Quality: MODERATE")
        else: st.error("Air Quality: UNHEALTHY")
    else:
        st.warning("Model not loaded.")

# --- ROW 3: ANALYTICS ---
st.divider()
st.subheader("📊 Data Analytics")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Temperature vs PM2.5**")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='temperature', y='pm2_5', alpha=0.5, ax=ax)
    ax.set_ylim(0, 100) # Limit y-axis to see typical range
    st.pyplot(fig)

with chart_col2:
    st.markdown("**PM2.5 Distribution**")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['pm2_5'], bins=30, kde=True, ax=ax2)
    ax2.set_xlim(0, 100)
    st.pyplot(fig2)