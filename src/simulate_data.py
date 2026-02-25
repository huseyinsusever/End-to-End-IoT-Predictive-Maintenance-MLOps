import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_sensor_data(num_machines=5, days=30, points_per_day=24):
    """
    Simulates IoT sensor data for multiple machines.
    Includes normal operation and degradation periods leading to failures.
    """
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=days)
    
    all_data = []
    
    for machine_id in range(1, num_machines + 1):
        # Machine specific baseline characteristics
        base_temp = np.random.uniform(60, 80)
        base_vib = np.random.uniform(1.0, 3.0)
        base_pressure = np.random.uniform(100, 120)
        
        # Decide if this machine will fail during the simulated period
        will_fail = np.random.choice([True, False], p=[0.4, 0.6])
        
        total_points = days * points_per_day
        
        # If it fails, pick a random point in the last 20% of the timeline
        failure_point = total_points if not will_fail else np.random.randint(int(total_points * 0.8), total_points)
        
        for step in range(total_points):
            current_time = start_date + timedelta(hours=step * (24/points_per_day))
            
            # Remaining Useful Life (RUL) in days. 
            # If not failing in this window, we just set a high default value.
            rul = max(0, (failure_point - step) / points_per_day) if will_fail else 50.0
            
            # Normal noise
            t_noise = np.random.normal(0, 2)
            v_noise = np.random.normal(0, 0.2)
            p_noise = np.random.normal(0, 5)
            
            # Default values
            curr_temp = base_temp + t_noise
            curr_vib = base_vib + v_noise
            curr_pressure = base_pressure + p_noise
            is_anomaly = 0
            
            # Introduce degradation as it approaches failure
            days_to_fail = (failure_point - step) / points_per_day
            
            if will_fail and days_to_fail < 5 and days_to_fail >= 0:
                # Degradation phase: metrics start trending upwards and become more volatile
                degradation_factor = (5 - days_to_fail) / 5.0 # Goes from 0 to 1
                curr_temp += degradation_factor * 15 + np.random.normal(0, 5 * degradation_factor)
                curr_vib += degradation_factor * 4 + np.random.normal(0, 1 * degradation_factor)
                curr_pressure -= degradation_factor * 20 + np.random.normal(0, 8 * degradation_factor)
                
                # Mark as anomaly if degradation is severe enough (e.g., last 2 days)
                if days_to_fail < 2:
                    is_anomaly = 1
                    
            # Random spikes (anomalies) even during normal operation (e.g. 1% chance)
            elif np.random.random() < 0.01:
                curr_temp += np.random.normal(20, 5)
                curr_vib += np.random.normal(5, 1)
                is_anomaly = 1

            # Stop generating data after failure
            if step > failure_point:
                break
                
            all_data.append({
                'timestamp': current_time,
                'machine_id': f'M_{machine_id:03d}',
                'temperature': round(curr_temp, 2),
                'vibration': round(curr_vib, 3),
                'pressure': round(curr_pressure, 2),
                'is_anomaly': is_anomaly,
                'rul_days': round(rul, 2)
            })
            
    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Generating synthetic IoT sensor data...")
    df = generate_sensor_data(num_machines=20, days=60, points_per_day=24) # 1 reading per hour
    
    # Ensure data directory exists
    data_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'sensor_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Data generation complete. Produced {len(df)} records.")
    print(f"Saved to: {output_path}")
    print("\nSample Data:")
    print(df.head())
    print("\nTarget Class Distribution (is_anomaly):")
    print(df['is_anomaly'].value_counts())
