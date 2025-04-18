import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# --- Simulated Data ---
years = np.arange(2000, 2021)
temperature = np.linspace(120, 145, len(years)) + np.random.normal(0, 1.5, len(years))
extraction_rate = np.linspace(50, 100, len(years)) + np.random.normal(0, 5, len(years))

df = pd.DataFrame({
    'Year': years,
    'Temperature': temperature,
    'ExtractionRate': extraction_rate
})

# --- Preprocessing ---
scaler_temp = MinMaxScaler()
scaler_ext = MinMaxScaler()

temp_scaled = scaler_temp.fit_transform(df['Temperature'].values.reshape(-1, 1))
ext_scaled = scaler_ext.fit_transform(df['ExtractionRate'].values.reshape(-1, 1))

X_temp, y_temp = [], []
X_ext, y_ext = [], []
window_size = 3

for i in range(window_size, len(temp_scaled)):
    X_temp.append(temp_scaled[i-window_size:i])
    y_temp.append(temp_scaled[i])
    X_ext.append(ext_scaled[i-window_size:i])
    y_ext.append(ext_scaled[i])

X_temp, y_temp = np.array(X_temp), np.array(y_temp)
X_ext, y_ext = np.array(X_ext), np.array(y_ext)

# --- LSTM Models ---
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X_temp.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model_temp = build_model()
model_ext = build_model()

model_temp.fit(X_temp, y_temp, epochs=100, verbose=0)
model_ext.fit(X_ext, y_ext, epochs=100, verbose=0)

# --- Forecast Future Values ---
future_years = np.arange(2021, 2031)
forecast_temp = []
forecast_ext = []

last_temp_seq = temp_scaled[-window_size:]
last_ext_seq = ext_scaled[-window_size:]

for _ in range(len(future_years)):
    pred_temp = model_temp.predict(last_temp_seq.reshape(1, window_size, 1), verbose=0)
    forecast_temp.append(pred_temp[0][0])
    last_temp_seq = np.append(last_temp_seq[1:], pred_temp).reshape(window_size, 1)

    pred_ext = model_ext.predict(last_ext_seq.reshape(1, window_size, 1), verbose=0)
    forecast_ext.append(pred_ext[0][0])
    last_ext_seq = np.append(last_ext_seq[1:], pred_ext).reshape(window_size, 1)

forecast_temp = scaler_temp.inverse_transform(np.array(forecast_temp).reshape(-1, 1)).flatten()
forecast_ext = scaler_ext.inverse_transform(np.array(forecast_ext).reshape(-1, 1)).flatten()

# --- Plot Combined Forecast ---
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_title('Geothermal Temperature & Extraction Rate Forecast in Bangladesh (2000–2030)', fontsize=14)
ax1.set_xlabel('Year', fontsize=12)

# Temperature on Left Y-axis
ax1.set_ylabel('Temperature (°C)', color='orange', fontsize=12)
ax1.plot(df['Year'], df['Temperature'], color='orange', marker='o', label='Observed Temp')
ax1.plot(future_years, forecast_temp, color='darkorange', linestyle='--', marker='x', label='Forecasted Temp')
ax1.tick_params(axis='y', labelcolor='orange')

# Extraction on Right Y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Extraction Rate (MW)', color='steelblue', fontsize=12)
ax2.plot(df['Year'], df['ExtractionRate'], color='steelblue', marker='s', label='Observed Extraction')
ax2.plot(future_years, forecast_ext, color='navy', linestyle='--', marker='^', label='Forecasted Extraction')
ax2.tick_params(axis='y', labelcolor='steelblue')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.grid(True)
plt.tight_layout()
plt.show()

