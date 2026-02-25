import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Lenovo IoT Predictive Maintenance", layout="wide")

st.title("🏭 IoT Predictive Maintenance Dashboard")
st.markdown("Real-time monitoring and anomaly detection for manufacturing equipment.")

# Load a sample of data to simulate streaming
@st.cache_data
def load_data():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(BASE_DIR, 'data', 'sensor_data.csv')
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Pick one machine to monitor for the demo
        machine_df = df[df['machine_id'] == 'M_001'].sort_values('timestamp')
        return machine_df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data found. Please run data generation and model training scripts first.")
    st.stop()

# Dashboard Layout
st.sidebar.header("Controls")
simulation_speed = st.sidebar.slider("Simulation Speed (ms delay)", 50, 1000, 200)
start_sim = st.sidebar.button("Start Real-time Simulation")
stop_sim = st.sidebar.button("Stop Simulation")

# Placeholders for dynamic content
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
temp_metric = metrics_col1.empty()
vib_metric = metrics_col2.empty()
rul_metric = metrics_col3.empty()

alert_placeholder = st.empty()

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Temperature Trend")
    temp_chart = st.empty()
with chart_col2:
    st.subheader("Vibration Trend")
    vib_chart = st.empty()


def send_prediction_request(row):
    """Sends current sensor reading to FastAPI backend"""
    # In a real scenario, this would hit http://localhost:8000/predict
    # For robust local demo without needing two terminals running simultaneously,
    # we can optionally do direct prediction here if API is down, 
    # but let's try the API first.
    api_url = "http://localhost:8000/predict"
    
    # We need to calculate matching rolling features for the request
    # In production, a streaming engine (Kafka/Flink) or the API itself might handle this states
    # For the UI demo, we send the pre-calculated features from our dataset
    
    payload = {
        "temperature": row['temperature'],
        "vibration": row['vibration'],
        "pressure": row['pressure'],
        "temperature_rolling_mean": row['temperature'] if 'temperature_rolling_mean' not in row else row['temperature_rolling_mean'],
        "temperature_rolling_std": 0.0 if 'temperature_rolling_std' not in row else row['temperature_rolling_std'],
        "vibration_rolling_mean": row['vibration'] if 'vibration_rolling_mean' not in row else row['vibration_rolling_mean'],
        "vibration_rolling_std": 0.0 if 'vibration_rolling_std' not in row else row['vibration_rolling_std'],
        "pressure_rolling_mean": row['pressure'] if 'pressure_rolling_mean' not in row else row['pressure_rolling_mean'],
        "pressure_rolling_std": 0.0 if 'pressure_rolling_std' not in row else row['pressure_rolling_std']
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass # API might be off
    return None

# State management for simulation
if 'simulate' not in st.session_state:
    st.session_state.simulate = False
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

if start_sim:
    st.session_state.simulate = True
if stop_sim:
    st.session_state.simulate = False

# We need to ensure we have the rolling features available in df
window = 12
df['temperature_rolling_mean'] = df['temperature'].rolling(window=window, min_periods=1).mean()
df['temperature_rolling_std'] = df['temperature'].rolling(window=window, min_periods=1).std().fillna(0)
df['vibration_rolling_mean'] = df['vibration'].rolling(window=window, min_periods=1).mean()
df['vibration_rolling_std'] = df['vibration'].rolling(window=window, min_periods=1).std().fillna(0)
df['pressure_rolling_mean'] = df['pressure'].rolling(window=window, min_periods=1).mean()
df['pressure_rolling_std'] = df['pressure'].rolling(window=window, min_periods=1).std().fillna(0)


if st.session_state.simulate:
    # Pre-allocate historical data for smooth charting
    history_size = 50
    
    for i in range(st.session_state.current_idx, len(df)):
        if not st.session_state.simulate:
            break
            
        current_row = df.iloc[i]
        
        # 1. Update Metrics
        temp_metric.metric("Temperature (°C)", f"{current_row['temperature']:.1f}")
        vib_metric.metric("Vibration (mm/s)", f"{current_row['vibration']:.2f}")
        
        # 2. Get Prediction from API
        prediction = send_prediction_request(current_row)
        
        # Display Prediction Results
        if prediction:
            is_anomaly = prediction['is_anomaly']
            rul = prediction['predicted_rul_days']
            
            if is_anomaly:
                alert_placeholder.error(f"🚨 ANOMALY DETECTED! Immediate maintenance recommended. System Score: {prediction['anomaly_score']:.2f}")
                rul_metric.metric("Predicted RUL (Days)", f"{rul:.1f}", delta="-CRITICAL", delta_color="inverse")
            else:
                alert_placeholder.success("✅ System Operating Normally")
                rul_metric.metric("Predicted RUL (Days)", f"{rul:.1f}", delta="Stable", delta_color="normal")
        else:
            alert_placeholder.warning("⚠️ Cannot connect to Prediction API. Ensure FastAPI is running on port 8000.")
            # Fallback to true labels if API is down for demo purposes
            rul_metric.metric("True RUL (Simulated Data)", f"{current_row['rul_days']:.1f}")
            if current_row['is_anomaly'] == 1:
                alert_placeholder.error(f"🚨 ANOMALY DETECTED (Offline Mode)")
            
        # 3. Update Charts
        start_idx = max(0, i - history_size)
        window_df = df.iloc[start_idx:i+1]
        
        fig_temp = px.line(window_df, x='timestamp', y='temperature', 
                           title="Live Temperature", template="plotly_dark")
        
        # Add anomaly markers to the chart if any happened in this window
        anomalies_window = window_df[window_df['is_anomaly'] == 1]
        if not anomalies_window.empty:
            fig_temp.add_trace(go.Scatter(x=anomalies_window['timestamp'], y=anomalies_window['temperature'],
                                          mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
            
        temp_chart.plotly_chart(fig_temp, use_container_width=True)
        
        fig_vib = px.line(window_df, x='timestamp', y='vibration', 
                          title="Live Vibration", template="plotly_dark", color_discrete_sequence=['orange'])
        if not anomalies_window.empty:
             fig_vib.add_trace(go.Scatter(x=anomalies_window['timestamp'], y=anomalies_window['vibration'],
                                          mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
             
        vib_chart.plotly_chart(fig_vib, use_container_width=True)
        
        st.session_state.current_idx = i + 1
        time.sleep(simulation_speed / 1000.0)
        
    if st.session_state.current_idx >= len(df):
        st.session_state.current_idx = 0
        st.session_state.simulate = False
        st.sidebar.success("Simulation finished.")
