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
* Trained a **Proximal Policy Optimization (PPO)** agent (`MlpPolicy`) via `Stable-Baselines3` over $50,000$ timesteps.
* Replaced iterative runtime evolutionary search ($O(\text{Generations} \times \text{Population})$) with a single neural network forward pass ($O(1)$ inference).

### Phase 5: Closed-Loop Hardware-in-the-Loop (`firmware/esp32_firmware.ino`)
* **Sensing ($t+1$ Forecast):** ESP32 polls ambient temperature and relative humidity via **DHT22** (`GPIO 15`) and room illuminance via an **LDR module** (`GPIO 34`). Python computes live Dew Point via the Magnus-Tetens approximation:
  $$T_d \approx T - \frac{100 - RH}{5}$$
* **Actuation (PWM):** Optimal setpoints (`ACT:24.1,50.0\n`) are translated into 8-bit PWM duty cycles (`0–255`) driving an **L298N H-Bridge Motor Driver + DC Fan** (`GPIO 4`) and a **Dimmable LED** (`GPIO 2`).

---

## 📊 Key Results & Benchmarks

| Metric | Unoptimized Baseline | Genetic Algorithm (Phase 3) | PPO RL Agent (Phase 4) |
| :--- | :---: | :---: | :---: |
| **HVAC Setpoint** | $21.0^\circ\text{C}$ | $24.1^\circ\text{C}$ | $24.1^\circ\text{C}$ |
| **Lighting Level** | $100.0\%$ | $50.0\%$ | $50.0\%$ |
| **Hourly Energy Load** | $169.85\text{ kWh}$ | $143.20\text{ kWh}$ | $143.22\text{ kWh}$ |
| **Energy Reduction** | Baseline | **15.7% Saved** | **15.7% Saved** |
| **Execution Latency** | N/A | $\sim 450 - 1200\text{ ms}$ | **$\sim 1.5 - 3.0\text{ ms}$** |
| **Time Complexity** | $O(1)$ Static | $O(G \times P)$ Search | **$O(1)$ Forward Pass** |

---

## 📂 Repository Structure

```text
AI-Building-Energy-Optimizer/
├── firmware/
│   └── esp32_firmware.ino               # C++ Arduino sketch for ESP32 sensor/PWM loop
├── models/
│   ├── xgboost_energy_predictor.pkl     # Trained 15-feature XGBoost Digital Twin
│   ├── feature_names.pkl                # Serialized feature column order
│   └── rl_energy_agent.zip              # Trained Stable-Baselines3 PPO policy
├── preprocess.py                        # Phase 1: Data cleaning & memory downcasting
├── train.py                             # Phase 2: Model training & benchmarking
├── 03_optimize.py                       # Phase 3: Standalone Differential Evolution CLI
├── 04_train_rl.py                       # Phase 4: Gymnasium env & PPO agent training
├── app.py                               # Phase 5 & 6: Multi-tab Streamlit + HIL dashboard
├── requirements.txt                     # Python package dependencies
├── .gitignore                           # Excludes raw multi-GB CSV/Parquet files
└── README.md                            # Project documentation
```

---

## 🔌 Hardware Pinout & Wiring (ESP32)

| Component | Pin on Module | ESP32 GPIO / Rail | Purpose |
| :--- | :---: | :---: | :--- |
| **DHT22 Sensor** | `VCC` / `DATA` / `GND` | `3V3` / `GPIO 15` / `GND` | Live room Temperature & Humidity |
| **LDR Sensor Module** | `VCC` / `AO` / `GND` | `3V3` / `GPIO 34` / `GND` | Ambient daylight / cloud estimation |
| **Room Lighting LED** | `Anode (+)` / `Cathode (-)` | `GPIO 2` / `GND` (via $220\Omega$) | PWM dimming visualization |
| **L298N Motor Driver** | `ENA` / `IN1` / `IN2` | `GPIO 4` / `3V3` / `GND` | HVAC DC fan speed control via PWM |

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/AI-Building-Energy-Optimizer.git
cd AI-Building-Energy-Optimizer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the RL Policy
```bash
python 04_train_rl.py
```

### 4. Launch the Interactive BEMS Dashboard
```bash
python -m streamlit run app.py
```
* **Simulation Mode:** Leave the *"Enable Physical ESP32 Bridge"* sidebar toggle **OFF** to test weather sliders, 24-hour load profiles, and the GA vs. RL benchmark tab purely in software.
* **Hardware-in-the-Loop Mode:** Flash `firmware/esp32_firmware.ino` to your ESP32, connect via USB, enter your COM port (e.g., `COM3`), and toggle the bridge **ON** to stream live sensor telemetry and actuate the physical fan and LED.

---

## 📜 License & Acknowledgments
* **Dataset:** ASHRAE — *Great Energy Predictor III* (hosted on Kaggle).
* Developed for academic evaluation under the **Innovation Design Project (IDP)** curriculum.
