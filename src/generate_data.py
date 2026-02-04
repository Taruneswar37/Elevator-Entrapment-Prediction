# =====================================================
# src/generate_data.py
# =====================================================

import os
import pandas as pd
import numpy as np

# ------------------- SEED -------------------
np.random.seed(42)

# ------------------- SAMPLES -------------------
n_samples = 15000

# ------------------- BASE SIGNALS -------------------
data = pd.DataFrame({
    'vibration_hz': np.random.normal(5, 2.5, n_samples).clip(0, 15),
    'usage_rate': np.random.poisson(60, n_samples).clip(0, 200),
    'door_cycles': np.random.poisson(150, n_samples).clip(0, 500),
    'speed_mps': np.random.normal(1.2, 0.2, n_samples).clip(0.5, 2.0),
    'temp_celsius': np.random.normal(40, 8, n_samples).clip(20, 60)
})

# ------------------- TEMPORAL FEATURES -------------------
data['vibration_trend'] = data['vibration_hz'].rolling(5).mean()
data['cycle_trend'] = data['door_cycles'].rolling(5).mean()
data['temp_trend'] = data['temp_celsius'].rolling(5).mean()

# Backfill first few rows
data = data.bfill()

# ------------------- REALISTIC FAILURE LOGIC -------------------
data['entrapment_risk'] = (
    (data['vibration_trend'] > 7.2) |
    (data['cycle_trend'] > 220) |
    (data['temp_trend'] > 55) |
    ((data['door_cycles'] > 250) & (data['usage_rate'] > 120))
).astype(int)

# ------------------- CLASS DISTRIBUTION -------------------
print("Class Distribution:")
print(data['entrapment_risk'].value_counts(normalize=True))

# ------------------- SAVE DATA -------------------
# Get folder paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
DATA_DIR = os.path.join(BASE_DIR, "../data")
os.makedirs(DATA_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, "elevator_data.csv")
data.to_csv(csv_path, index=False)

print("✅ Dataset ready with temporal patterns!")
print("Saved at:", csv_path)
