import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import random
import statsmodels.api as sm

# ------------------------
# Plot 1: Reinforcement Learning for Geothermal Site Sustainability
# ------------------------
episodes = 100
reward = []
resource = 1000
for i in range(episodes):
    extract = random.randint(5, 15)
    resource -= extract * 0.95  # with 5% natural replenishment
    r = extract if resource > 0 else -10
    reward.append(r + (resource * 0.001))

# ------------------------
# Plot 2: LSTM Forecast for Temperature Profile
# ------------------------
temperature = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)
scaler = MinMaxScaler()
temperature_scaled = scaler.fit_transform(temperature.reshape(-1, 1))

seq_len = 10
X, y = [], []
for i in range(len(temperature_scaled) - seq_len):
    X.append(temperature_scaled[i:i + seq_len])
    y.append(temperature_scaled[i + seq_len])
X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(50, input_shape=(seq_len, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, verbose=0)

forecast_input = temperature_scaled[-seq_len:].reshape(1, seq_len, 1)
forecast = model.predict(forecast_input, verbose=0)
forecast = scaler.inverse_transform(forecast)[0]

# ------------------------
# Plot 3: ARIMA Grid Stability Prediction
# ------------------------
grid_stability = np.random.normal(1.0, 0.05, 120)
model_arima = sm.tsa.ARIMA(grid_stability[:100], order=(2, 1, 2))
fit_arima = model_arima.fit()
forecast_arima = fit_arima.forecast(steps=20)

# ------------------------
# Plot 4: Heatmap for Power Grid Load
# ------------------------
grid_matrix = np.random.uniform(50, 100, (10, 10))

# --------- Create 2x2 Subplot Grid with Reduced Size ---------
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# Plot 1: Reinforcement Learning Reward
axs[0, 0].plot(range(episodes), reward, color='teal', marker='o')
axs[0, 0].set_title("Plot 1: RL Reward for Sustainable Extraction")
axs[0, 0].set_xlabel("Episode")
axs[0, 0].set_ylabel("Cumulative Reward")

# Plot 2: LSTM Forecast
axs[0, 1].plot(range(seq_len, seq_len + len(forecast)), forecast, label='LSTM Forecast', color='purple')
axs[0, 1].plot(range(len(temperature)), temperature, label='Actual', linestyle='dashed', alpha=0.6)
axs[0, 1].set_title("Plot 2: LSTM Forecast of Temperature")
axs[0, 1].set_xlabel("Time Step")
axs[0, 1].set_ylabel("Temperature (°C)")
axs[0, 1].legend()

# Plot 3: ARIMA Grid Stability Forecast
axs[1, 0].plot(range(100), grid_stability[:100], label='Original Stability', alpha=0.7)
axs[1, 0].plot(range(100, 120), forecast_arima, label='Forecast Stability', color='darkred')
axs[1, 0].set_title("Plot 3: ARIMA Grid Stability Forecast")
axs[1, 0].set_xlabel("Time Step")
axs[1, 0].set_ylabel("Stability Index")
axs[1, 0].legend()

# Plot 4: Grid Load Balance Heatmap
sns.heatmap(grid_matrix, cmap='coolwarm', ax=axs[1, 1], cbar_kws={'label': 'Load %'})
axs[1, 1].set_title("Plot 4: Grid Load Balance Simulation")
axs[1, 1].set_xlabel("Region")
axs[1, 1].set_ylabel("Time Window")

plt.tight_layout()
plt.show()