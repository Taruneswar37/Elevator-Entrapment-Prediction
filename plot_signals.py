import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your dataset
data = pd.read_csv("data/elevator_data.csv")   # <-- use your actual file name

# 2. Create figure with 5 subplots
plt.figure(figsize=(12, 10))

# ----- VIBRATION -----
plt.subplot(5, 1, 1)
plt.plot(data['vibration_hz'])
plt.title("Vibration (Hz)")
plt.ylabel("Hz")

# ----- TEMPERATURE -----
plt.subplot(5, 1, 2)
plt.plot(data['temp_celsius'])
plt.title("Temperature (°C)")
plt.ylabel("°C")

# ----- DOOR CYCLES -----
plt.subplot(5, 1, 3)
plt.plot(data['door_cycles'])
plt.title("Door Cycles")
plt.ylabel("cycles")

# ----- USAGE RATE -----
plt.subplot(5, 1, 4)
plt.plot(data['usage_rate'])
plt.title("Usage Rate")
plt.ylabel("count")

# ----- SPEED -----
plt.subplot(5, 1, 5)
plt.plot(data['speed_mps'])
plt.title("Cabin Speed (m/s)")
plt.ylabel("m/s")
plt.xlabel("Time Steps")

plt.tight_layout()

# 3. Save exactly with the name we use in paper
plt.savefig("fig1_signals.png", dpi=300)
plt.show()