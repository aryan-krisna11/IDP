import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Load the preprocessed data directly from disk
print("Loading preprocessed dataset...")
df = pd.read_parquet("data/processed/ashrae_cleaned.parquet")

# 2. Chronological Train-Validation Split (80% train, 20% test)
split_point = df["timestamp"].quantile(0.80)
train_mask = df["timestamp"] < split_point
val_mask = df["timestamp"] >= split_point

feature_cols = [c for c in df.columns if c not in ["target", "timestamp"]]

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, "target"]

X_val = df.loc[val_mask, feature_cols]
y_val = df.loc[val_mask, "target"]

print(f"Training rows: {len(X_train):,} | Validation rows: {len(X_val):,}")

# 3. Optional: Subsample training set if fitting takes too long on local machine
if len(X_train) > 300000:
    print("Subsampling 300,000 rows from training set for quick model fitting...")
    sample_idx = X_train.sample(n=300000, random_state=42).index
    X_train_fit = X_train.loc[sample_idx]
    y_train_fit = y_train.loc[sample_idx]
else:
    X_train_fit = X_train
    y_train_fit = y_train

# 4. Define Phase 2 Models
models = {
    "Linear Regression (Baseline)": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
    "XGBoost (Main Model)": xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42
    )
}

results = []

# 5. Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_fit, y_train_fit)
    
    # Predict (output is in log1p scale)
    preds_log = model.predict(X_val)
    
    # Invert log1p back to actual kWh: exp(y) - 1
    preds_kwh = np.expm1(preds_log)
    actual_kwh = np.expm1(y_val)
    
    mae = mean_absolute_error(actual_kwh, preds_kwh)
    rmse = root_mean_squared_error(actual_kwh, preds_kwh)
    r2 = r2_score(actual_kwh, preds_kwh)
    
    results.append({
        "Model": name,
        "MAE (kWh)": round(mae, 3),
        "RMSE (kWh)": round(rmse, 3),
        "R² Score": round(r2, 4)
    })

# Display Phase 2 Evaluation Table
print("\n--- Phase 2 Model Evaluation Comparison ---")
print(pd.DataFrame(results).to_string(index=False))

# 6. Save the best model artifact (XGBoost) for Phase 3 and Phase 6
os.makedirs("models", exist_ok=True)
best_model = models["XGBoost (Main Model)"]
joblib.dump(best_model, "models/xgboost_energy_predictor.pkl")
joblib.dump(feature_cols, "models/feature_names.pkl")
print("\nSuccessfully saved best model to 'models/xgboost_energy_predictor.pkl'!")