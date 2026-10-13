import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric datatypes to drastically minimize memory footprint."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Use Pandas API type checking to safely handle StringDtype and objects.
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
                    
        elif pd.api.types.is_float_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Mem usage decreased to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def preprocess_ashrae_data(
    train_path="data/train.csv",
    building_path="data/building_metadata.csv",
    weather_path="data/weather_train.csv",
):
    print("Step 1: Reading and downcasting CSVs...")
    train_df = reduce_mem_usage(pd.read_csv(train_path))
    building_df = reduce_mem_usage(pd.read_csv(building_path))
    weather_df = reduce_mem_usage(pd.read_csv(weather_path))

    # Convert timestamps to datetime objects
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    print("Step 2: Merging data tables...")
    df = train_df.merge(building_df, on="building_id", how="left")
    df = df.merge(weather_df, on=["site_id", "timestamp"], how="left")
    del train_df, building_df, weather_df  # Free memory

    print("Step 3: Filtering & fixing known dataset anomalies...")
    # Focus on electricity meter (meter == 0) as primary target
    df = df[df["meter"] == 0].copy()

    # Site 0 electricity readings were recorded in kBTU instead of kWh; convert to kWh
    df.loc[df["site_id"] == 0, "meter_reading"] = (
        df.loc[df["site_id"] == 0, "meter_reading"] * 0.2931
    )

    # Remove zero-reading streaks (unmetered/offline instrumentation errors)
    # Filter out records where meter_reading == 0 during active months
    df = df[df["meter_reading"] > 0].copy()

    print("Step 4: Imputing missing weather data...")
    weather_cols = [
        "air_temperature",
        "dew_temperature",
        "wind_speed",
        "cloud_coverage",
    ]
    # Forward and backward fill weather variables per site chronologically
    df.sort_values(by=["site_id", "timestamp"], inplace=True)
    for col in weather_cols:
        df[col] = df.groupby("site_id")[col].transform(
            lambda x: x.ffill().bfill()
        )
        # In case a site has entire series missing, fill with overall mean
        df[col] = df[col].fillna(df[col].median())

    # Metadata imputation
    df["floor_count"] = df["floor_count"].fillna(1)
    df["year_built"] = df["year_built"].fillna(df["year_built"].median())

    print("Step 5: Feature Engineering...")
    # Time signals
    df["hour"] = df["timestamp"].dt.hour.astype(np.uint8)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(np.uint8)
    df["month"] = df["timestamp"].dt.month.astype(np.uint8)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.uint8)

    # Cyclical encodings for periodic time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(np.float32)

    # Categorical encoding for primary building function
    df["primary_use"] = df["primary_use"].astype("category").cat.codes

    # Target transform: log1p dampens extreme skewness
    df["target"] = np.log1p(df["meter_reading"]).astype(np.float32)

    print("Step 6: Finalizing feature set...")
    feature_cols = [
        "building_id",
        "site_id",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "air_temperature",
        "dew_temperature",
        "wind_speed",
        "cloud_coverage",
        "hour_sin",
        "hour_cos",
        "dayofweek",
        "month",
        "is_weekend",
    ]

    return df[feature_cols], df["target"], df["timestamp"]


# --- Run Script ---
if __name__ == "__main__":
    X, y, timestamps = preprocess_ashrae_data()
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Combine into a single clean DataFrame to store on disk
    processed_df = X.copy()
    processed_df["target"] = y
    processed_df["timestamp"] = timestamps

    # Create a processed data directory if it doesn't exist
    import os
    os.makedirs("data/processed", exist_ok=True)

    print("Saving processed data to disk...")
    # Parquet preserves exact data types and loads in seconds
    processed_df.to_parquet("data/processed/ashrae_cleaned.parquet", index=False)
    print("Saved to data/processed/ashrae_cleaned.parquet successfully!")
