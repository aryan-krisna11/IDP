# ⚡ Intelligent Building Energy Management System (BEMS)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Digital_Twin-XGBoost-orange)
![Stable-Baselines3](https://img.shields.io/badge/Reinforcement_Learning-PPO-success)
![SciPy](https://img.shields.io/badge/Metaheuristics-Differential_Evolution-blueviolet)
![ESP32](https://img.shields.io/badge/Hardware_HIL-ESP32_IoT-red?logo=espressif&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

An end-to-end **Cyber-Physical Building Energy Optimization System** developed as an academic **Innovation Design Project (IDP)**. This system combines a high-speed **XGBoost Digital Twin**, **Metaheuristic Evolutionary Search (Differential Evolution)**, **Deep Reinforcement Learning (Proximal Policy Optimization)**, and a bidirectional **ESP32 Hardware-in-the-Loop (HIL)** testbed to dynamically minimize commercial building electricity consumption while strictly preserving human thermal and visual comfort.

---

## 👥 Project Team
* **Siddhant Singh** — `25BAI1506`
* **Aryan Krisna** — `25BAI1058`
* **Anant Tripathi** — `25BAI1464`

---

## 🎯 Problem Statement & Motivation
Commercial buildings account for nearly **30–40% of global energy demand**, with Heating, Ventilation, and Air Conditioning (HVAC) and interior lighting representing the largest controllable loads. Traditional Building Management Systems (BMS) rely on static, rule-based setpoints (e.g., locking AC at $21.0^\circ\text{C}$ and lighting at $100\%$) regardless of ambient weather fluctuations or occupancy thermodynamics. 

**Our Solution:** We transition building operations from static rules to **prescriptive, closed-loop AI control**:
1. **Forecast** next-hour power demand using real-time environmental telemetry.
2. **Optimize** HVAC and lighting setpoints via multi-objective cost-comfort functions.
3. **Actuate** physical hardware in milliseconds using an $O(1)$ Deep Reinforcement Learning policy.

---

## 🏗️ 6-Phase System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1 & 2: DATA & DIGITAL TWIN                          │
│  ASHRAE GEPIII Dataset ──> Memory Downcasting & Cyclical Time Encoding      │
│                        ──> XGBoost Surrogate Predictor (MAE: ~39.99 kWh)    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 3 & 4: PRESCRIPTIVE AI CONTROL                      │
│  • Engine A (Phase 3): Differential Evolution (Genetic Algorithm Search)    │
│  • Engine B (Phase 4): PPO Deep RL Agent in Custom Gymnasium Environment    │
│    (Reduces inference latency from ~1200 ms search to ~2 ms O(1) policy)    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                        USB UART Serial (115200 bps)
                        "READ\n"  <──────>  "ACT:24.1,50.0\n"
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 5 & 6: HIL & STREAMLIT EMS                          │
│  • ESP32 Microcontroller: Reads DHT22 (Temp/RH) & LDR (Ambient Lux)         │
│  • PWM Actuation: Drives L298N HVAC DC Fan (GPIO 4) & Room LED (GPIO 2)     │
│  • Multi-Tab Streamlit UI: Live Telemetry, 24h Load Curve & Carbon ROI      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Methodology

### Phase 1: Data Engineering (`preprocess.py`)
* **Dataset:** ASHRAE Great Energy Predictor III (GEPIII).
* **Memory Optimization:** Downcasted 64-bit numeric columns to `int8`, `uint16`, and `float32`, reducing RAM footprint by **>60%** and serializing processed tables to `.parquet`.
* **Feature Engineering:** 
  * Target stabilization using log-transformation: $y_{\text{norm}} = \log(1 + \text{meter\_reading})$.
  * Cyclical encoding of 24-hour time continuity:
    $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$

### Phase 2: Surrogate Digital Twin (`train.py`)
* Benchmarked Linear Regression and Random Forest against **XGBoost** (`tree_method="hist"`).
* Trained on **15 core features** encompassing building metadata (`square_feet`, `floor_count`, `year_built`) and meteorological drivers (`air_temperature`, `dew_temperature`, `wind_speed`, `cloud_coverage`).
* Achieved a validation **Mean Absolute Error (MAE) of ~39.99 kWh**, serving as the physics simulator for downstream control engines.

### Phase 3: Metaheuristic Optimization (`03_optimize.py`)
* Formulated a constrained multi-objective loss function balancing energy load ($E$), tariff rate ($R$), and quadratic occupant discomfort penalties:
  $$\mathcal{L}(T_{\text{ac}}, L_{\%}) = w_1 E(T_{\text{ac}}, L_{\%}) + w_2 (E \cdot R) + w_3 \left[ (T_{\text{ac}} - T_{\text{target}})^2 + \max(0, L_{\min} - L_{\%})^2 \right]$$
* Solved via `scipy.optimize.differential_evolution` across bounds $T_{\text{ac}} \in [20^\circ\text{C}, 26^\circ\text{C}]$ and $L_{\%} \in [20\%, 100\%]$.

### Phase 4: Deep Reinforcement Learning (`04_train_rl.py`)
* Wrapped the 15-feature XGBoost Digital Twin inside a custom **Gymnasium** environment (`EnergyEnv`) with continuous `spaces.Box` action and observation spaces.
* Trained a **Proximal Policy Optimization (PPO)** agent (`MlpPolicy`) via `Stable-
