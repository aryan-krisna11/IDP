import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# 1. Load trained model and features
model = joblib.load("models/xgboost_energy_predictor.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# 2. Define baseline contextual conditions (example: afternoon in summer)
context = {
    "building_id": 100,
    "site_id": 0,
    "primary_use": 0,       # Education
    "square_feet": 50000,
    "year_built": 2005,
    "floor_count": 3,
    "air_temperature": 32.0, # Hot outside (32°C)
    "dew_temperature": 21.0,
    "wind_speed": 3.5,
    "cloud_coverage": 2.0,
    "hour_sin": np.sin(2 * np.pi * 14 / 24.0),
    "hour_cos": np.cos(2 * np.pi * 14 / 24.0),
    "dayofweek": 2,          # Wednesday
    "month": 7,
    "is_weekend": 0
}

# 3. Comfort parameters
COMFORT_TEMP_TARGET = 23.5   # Ideal temperature (°C)
MIN_LUX_REQUIRED = 300       # Target indoor lighting lux
ELECTRICITY_RATE = 8.5       # Cost in ₹/kWh

def simulate_energy(ac_temp, light_level, base_context):
    """
    Uses the trained ML model + physical adjustments to simulate load.
    - Setting AC cooler increases energy demand.
    - Dimming lights saves direct lighting wattage.
    """
    row = pd.DataFrame([base_context])[feature_names]
    
    # Baseline predicted energy from XGBoost
    base_log_pred = model.predict(row)[0]
    base_kwh = float(np.expm1(base_log_pred))
    
    # Physical delta approximation:
    # ~6-8% energy increase per degree cooling below outdoor air
    # Lighting: scaled proportionally from a 15 kW lighting base load
    temp_delta_factor = 1.0 + 0.07 * (26.0 - ac_temp)
    light_kwh = 15.0 * (light_level / 100.0)
    
    total_energy = (base_kwh * 0.7 * temp_delta_factor) + (base_kwh * 0.3) + light_kwh
    return total_energy

def objective_function(variables):
    """
    Objective: Minimize [Energy + Electricity Cost + Comfort Penalty]
    variables[0] = AC setpoint temperature (20°C to 26°C)
    variables[1] = Lighting percentage (20% to 100%)
    """
    ac_temp, light_pct = variables
    
    # 1. Energy & Cost
    energy_kwh = simulate_energy(ac_temp, light_pct, context)
    monetary_cost = energy_kwh * ELECTRICITY_RATE
    
    # 2. Comfort penalties (Quadratic penalty for drifting away from comfort targets)
    temp_discomfort = (ac_temp - COMFORT_TEMP_TARGET) ** 2
    
    # Lighting penalty if dropped below acceptable threshold (50%)
    light_discomfort = max(0, 50.0 - light_pct) ** 2
    
    # Weight factors
    alpha = 1.0   # Energy weight
    beta = 0.5    # Cost weight
    gamma = 25.0  # Occupant comfort weight (high priority)
    
    loss = (alpha * energy_kwh) + (beta * monetary_cost) + (gamma * (temp_discomfort + light_discomfort))
    return loss

# 4. Define bounds: AC between 20°C and 26°C; Lights between 20% and 100%
bounds = [(20.0, 26.0), (20.0, 100.0)]

print("Running Genetic Optimization (Phase 3)...")
result = differential_evolution(objective_function, bounds, seed=42, maxiter=50)

best_temp, best_light = result.x
baseline_energy = simulate_energy(21.0, 100.0, context)  # Naive default: AC at 21°C, Lights at 100%
optimized_energy = simulate_energy(best_temp, best_light, context)
savings_pct = 100 * (baseline_energy - optimized_energy) / baseline_energy

print("\n--- Phase 3 Optimization Results ---")
print(f"Default Baseline Energy : {baseline_energy:.2f} kWh")
print(f"Optimized Energy        : {optimized_energy:.2f} kWh")
print(f"Energy Reduction        : {savings_pct:.1f}%")
print(f"Optimal AC Setpoint     : {best_temp:.1f} °C")
print(f"Optimal Lighting Level  : {best_light:.1f} %")