import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from stable_baselines3 import PPO

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Energy Optimizer", layout="wide")
st.title("⚡ AI-Based Building Energy Optimizer")
st.markdown("Predicting and optimizing HVAC and lighting to reduce electricity consumption while preserving comfort.")

# --- 2. Load Model & Features (Cached for speed) ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/xgboost_energy_predictor.pkl")
    features = joblib.load("models/feature_names.pkl")
    # <-- CHANGED: Load the trained RL Agent
    rl_agent = PPO.load("models/rl_energy_agent") 
    return model, features, rl_agent

model, feature_names, rl_agent = load_assets()

# --- 3. Sidebar User Inputs (Environment) ---
st.sidebar.header("🌍 Current Environment")
out_temp = st.sidebar.slider("Outdoor Temperature (°C)", 15.0, 45.0, 32.0)
out_humidity = st.sidebar.slider("Dew Temperature (°C)", 10.0, 30.0, 21.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 3.5)
electricity_rate = st.sidebar.number_input("Electricity Rate (₹/kWh)", value=8.5)

st.sidebar.header("⚙️ Comfort Targets")
target_temp = st.sidebar.number_input("Target Indoor Temp (°C)", value=23.5)
min_light = st.sidebar.slider("Minimum Lighting %", 20, 100, 50)

# Build context from inputs (using fixed metadata for a standard building)
context = {
    "building_id": 100,
    "site_id": 0,
    "primary_use": 0,
    "square_feet": 50000,
    "year_built": 2005,
    "floor_count": 3,
    "air_temperature": out_temp,
    "dew_temperature": out_humidity,
    "wind_speed": wind_speed,
    "cloud_coverage": 2.0,
    "hour_sin": np.sin(2 * np.pi * 14 / 24.0),
    "hour_cos": np.cos(2 * np.pi * 14 / 24.0),
    "dayofweek": 2,
    "month": 7,
    "is_weekend": 0
}

# --- 4. Core Logic (Prediction & Optimization) ---
def simulate_energy(ac_temp, light_level, current_context):
    row = pd.DataFrame([current_context])[feature_names]
    base_log_pred = model.predict(row)[0]
    base_kwh = float(np.expm1(base_log_pred))
    
    # Physics approximation scaling
    temp_delta_factor = 1.0 + 0.07 * (26.0 - ac_temp)
    light_kwh = 15.0 * (light_level / 100.0)
    return (base_kwh * 0.7 * temp_delta_factor) + (base_kwh * 0.3) + light_kwh

# Calculate Baseline (Naive AC at 21°C, Lights 100%)
baseline_energy = simulate_energy(21.0, 100.0, context)
baseline_cost = baseline_energy * electricity_rate

# --- 5. Main Dashboard UI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Baseline")
    st.metric("Baseline Energy", f"{baseline_energy:.2f} kWh")
    st.metric("Baseline Cost", f"₹ {baseline_cost:.2f}")
    st.markdown("**Settings:** AC @ 21.0°C | Lights @ 100%")
    
    run_optimization = st.button("🚀 Run AI Optimizer", use_container_width=True)

if run_optimization:
    with st.spinner("RL Agent applying optimal control policy..."):
        
        # 1. Format what the RL agent needs to see (the Observation Space)
        current_weather = np.array([out_temp, out_humidity, wind_speed], dtype=np.float32)
        
        # 2. The RL agent instantly predicts the best action based on its training
        optimal_action, _states = rl_agent.predict(current_weather)
        opt_temp, opt_light = optimal_action
        
        # 3. Pass those outputs to the simulator to get the energy usage
        opt_energy = simulate_energy(opt_temp, opt_light, context)
        opt_cost = opt_energy * electricity_rate
        
        energy_saved = baseline_energy - opt_energy
        savings_pct = (energy_saved / baseline_energy) * 100

    with col2:
        st.subheader("Optimized Results")
        
        # Display Top-level Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimized Energy", f"{opt_energy:.2f} kWh", f"-{energy_saved:.2f} kWh")
        m2.metric("Optimized Cost", f"₹ {opt_cost:.2f}", f"-₹ {baseline_cost - opt_cost:.2f}")
        m3.metric("Energy Saved", f"{savings_pct:.1f}%")
        
        # Display New AI-Driven Controls
        st.markdown("### 🤖 Recommended AI Controls")
        c1, c2 = st.columns(2)
        c1.info(f"**HVAC Setpoint:** {opt_temp:.1f} °C")
        c2.warning(f"**Lighting Level:** {opt_light:.1f} %")
        
        # Visualization: Plotly Bar Chart
        fig = go.Figure(data=[
            go.Bar(name='Baseline', x=['Energy (kWh)', 'Cost (₹)'], y=[baseline_energy, baseline_cost], marker_color='red'),
            go.Bar(name='Optimized', x=['Energy (kWh)', 'Cost (₹)'], y=[opt_energy, opt_cost], marker_color='green')
        ])
        fig.update_layout(barmode='group', title="Comparison: Baseline vs. Optimized", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)