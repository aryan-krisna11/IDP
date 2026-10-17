import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
from stable_baselines3 import PPO

class EnergyEnv(gym.Env):
    def __init__(self):
        super(EnergyEnv, self).__init__()
        
        # Load BOTH the model and the 15 feature names
        self.xgboost_model = joblib.load("models/xgboost_energy_predictor.pkl")
        self.feature_names = joblib.load("models/feature_names.pkl")
        
        # Action Space: [AC Temp, Lighting %]
        self.action_space = spaces.Box(
            low=np.array([20.0, 20.0]), 
            high=np.array([26.0, 100.0]), 
            dtype=np.float32
        )
        
        # Observation Space: [Outdoor Temp, Dew Temp, Wind Speed]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([50.0, 30.0, 20.0]), 
            dtype=np.float32
        )
        
        self.state = None
        
        # Base context containing the required 15 features
        self.base_context = {
            "building_id": 100,
            "site_id": 0,
            "primary_use": 0,
            "square_feet": 50000,
            "year_built": 2005,
            "floor_count": 3,
            "air_temperature": 32.0,
            "dew_temperature": 21.0,
            "wind_speed": 3.5,
            "cloud_coverage": 2.0,
            "hour_sin": np.sin(2 * np.pi * 14 / 24.0),
            "hour_cos": np.cos(2 * np.pi * 14 / 24.0),
            "dayofweek": 2,
            "month": 7,
            "is_weekend": 0
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Starting weather state: [32.0 Temp, 21.0 Dew, 3.5 Wind]
        self.state = np.array([32.0, 21.0, 3.5], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        ac_temp, light_pct = action
        
        # 1. Update the base context with the RL Agent's current observed weather
        current_context = self.base_context.copy()
        current_context["air_temperature"] = self.state[0]
        current_context["dew_temperature"] = self.state[1]
        current_context["wind_speed"] = self.state[2]
        
        # 2. Format it into a 15-column DataFrame so XGBoost doesn't crash
        row = pd.DataFrame([current_context])[self.feature_names]
        
        # 3. Predict Base Energy
        base_log_pred = self.xgboost_model.predict(row)[0]
        base_kwh = float(np.expm1(base_log_pred))
        
        # 4. Calculate Physics Scaling
        temp_factor = 1.0 + 0.07 * (26.0 - ac_temp)
        energy_kwh = (base_kwh * 0.7 * temp_factor) + (base_kwh * 0.3) + (15.0 * (light_pct / 100.0))
        
        # 5. Reward Function
        cost = energy_kwh * 8.50
        comfort_penalty = 25.0 * ((ac_temp - 23.5) ** 2)
        reward = -(cost + comfort_penalty)
        
        done = True 
        return self.state, reward, done, False, {}

# --- Training Execution ---
if __name__ == "__main__":
    env = EnergyEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Training AI Control Agent...")
    model.learn(total_timesteps=50000)
    
    model.save("models/rl_energy_agent")
    print("Agent saved successfully.")