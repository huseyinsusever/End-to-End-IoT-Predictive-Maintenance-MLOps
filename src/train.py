import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    """
    Loads raw sensor data and applies feature engineering.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by machine and time to calculate rolling features correctly
    df = df.sort_values(by=['machine_id', 'timestamp'])
    
    print("Applying Feature Engineering...")
    # Calculate rolling means and standard deviations for a 12-hour window (12 points)
    window_size = 12
    
    features_to_roll = ['temperature', 'vibration', 'pressure']
    
    for feature in features_to_roll:
        df[f'{feature}_rolling_mean'] = df.groupby('machine_id')[feature].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        df[f'{feature}_rolling_std'] = df.groupby('machine_id')[feature].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std().fillna(0)
        )
        
    # Drop rows with NaNs if any were created (though min_periods=1 prevents this mostly)
    df = df.dropna()
    return df

def train_anomaly_model(df, features):
    """
    Trains an Isolation Forest model to detect anomalies.
    Unsupervised learning - we don't use the 'is_anomaly' label for training, 
    but we can use it for evaluation.
    """
    print("\n--- Training Anomaly Detection Model (Isolation Forest) ---")
    X = df[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model (contamination is the expected proportion of outliers)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)
    
    # Evaluate conceptually (using our simulated labels)
    predictions = model.predict(X_scaled)
    # IsolationForest outputs -1 for anomaly, 1 for normal. Let's map to 1 and 0.
    predictions_mapped = [1 if p == -1 else 0 for p in predictions]
    
    df['predicted_anomaly'] = predictions_mapped
    
    print("Anomaly Detection Evaluation (Unsupervised vs Simulated Labels):")
    print(classification_report(df['is_anomaly'], df['predicted_anomaly']))
    
    return model, scaler

def train_rul_model(df, features):
    """
    Trains a Random Forest Regressor to predict Remaining Useful Life (RUL).
    Supervised learning using 'rul_days' as the target.
    """
    print("\n--- Training Predictive Maintenance Model (Random Forest) ---")
    
    # Filter data: We might only want to train RUL on machines that actually failed
    # or periods where degradation is starting. For simplicity, we train on all.
    X = df[features]
    y = df['rul_days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We can reuse the same scaler or a new one. Data is tree-based so scaling isn't strict, 
    # but good practice if combining models later. Let's just use raw features for RF.
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"RUL Model Evaluation (RMSE): {rmse:.2f} days")
    
    return model

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'sensor_data.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run simulate_data.py first.")
        exit(1)
        
    df = load_and_preprocess_data(data_path)
    
    # Define features to use for modeling
    feature_cols = [
        'temperature', 'vibration', 'pressure',
        'temperature_rolling_mean', 'temperature_rolling_std',
        'vibration_rolling_mean', 'vibration_rolling_std',
        'pressure_rolling_mean', 'pressure_rolling_std'
    ]
    
    # 1. Train Anomaly Model
    anomaly_model, scaler = train_anomaly_model(df, feature_cols)
    
    # 2. Train RUL Model
    rul_model = train_rul_model(df, feature_cols)
    
    # 3. Save Models and artifacts
    print("\nSaving models...")
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(anomaly_model, os.path.join(models_dir, 'anomaly_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'feature_scaler.joblib')) # Needed for API
    joblib.dump(rul_model, os.path.join(models_dir, 'rul_model.joblib'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.joblib')) # To know exact order
    
    print("Models saved successfully in 'models/' directory.")
