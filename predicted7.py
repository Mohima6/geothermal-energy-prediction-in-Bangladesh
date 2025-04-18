#national grid

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Simulate time steps
time_steps = np.arange(1, 101)

# Synthetic data (replace with real data)
load = np.sin(time_steps / 10) + np.random.normal(0, 0.1, 100)
power_gen = np.cos(time_steps / 15) + np.random.normal(0, 0.1, 100)
energy_dist = np.random.normal(50, 5, 100)
grid_stability = np.random.normal(0, 1, 100)
power_fluctuations = np.random.normal(0, 0.5, 100)
cost_optimization = 100 - (energy_dist - np.random.normal(0, 2, 100))  # Simulated cost data

# Normalize load and power_gen
scaler = MinMaxScaler(feature_range=(0, 1))
load_scaled = scaler.fit_transform(load.reshape(-1, 1))
power_gen_scaled = scaler.fit_transform(power_gen.reshape(-1, 1))

# Prepare data for LSTM
train_size = int(len(load_scaled) * 0.8)
train_load, test_load = load_scaled[:train_size], load_scaled[train_size:]

# Reshape for LSTM
train_load = train_load.reshape(train_load.shape[0], 1, 1)
test_load = test_load.reshape(test_load.shape[0], 1, 1)

# Build and train LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_load, train_load, epochs=10, batch_size=32, verbose=0)

# Predict load
load_forecast = model.predict(test_load)
load_forecast = scaler.inverse_transform(load_forecast)

# Create 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Power Generation Forecast
axs[0, 0].plot(time_steps, power_gen, label='Power Generation')
axs[0, 0].set_title('Time-Series Forecasting for Power Generation')
axs[0, 0].legend()

# Plot 2: Load Forecasting
axs[0, 1].plot(time_steps[-len(load_forecast):], load_forecast, label='Predicted Load')
axs[0, 1].plot(time_steps, load, label='Actual Load', alpha=0.5)
axs[0, 1].set_title('Load Forecasting')
axs[0, 1].legend()

# Plot 3: Energy Distribution Matching
axs[0, 2].plot(time_steps, energy_dist, label='Energy Distribution')
axs[0, 2].set_title('Energy Distribution Matching')
axs[0, 2].legend()

# Plot 4: Grid Stability Prediction
axs[1, 0].plot(time_steps, grid_stability, label='Grid Stability')
axs[1, 0].set_title('Grid Stability Prediction')
axs[1, 0].legend()

# Plot 5: Power Fluctuation Simulation
axs[1, 1].plot(time_steps, power_fluctuations, label='Power Fluctuations', color='red')
axs[1, 1].set_title('Simulation for Power Fluctuations')
axs[1, 1].legend()

# ✅ Plot 6: Cost-Optimization for Energy Distribution
axs[1, 2].plot(time_steps, cost_optimization, label='Cost Optimization', color='green')
axs[1, 2].set_title('Cost-Optimization for Energy Distribution')
axs[1, 2].legend()

# Improve layout
plt.tight_layout()
plt.show()
