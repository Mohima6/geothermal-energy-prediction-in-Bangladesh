import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# ========== 1. Seismic Activity Forecasting with LSTM ========== #
np.random.seed(42)
time_steps = 100
seismic_series = np.sin(np.linspace(0, 10, time_steps)) + np.random.normal(0, 0.1, time_steps)

# Time-series preparation
X_seq, y_seq = [], []
for i in range(10, time_steps):
    X_seq.append(seismic_series[i-10:i])
    y_seq.append(seismic_series[i])
X_seq = np.array(X_seq).reshape(-1, 10, 1)
y_seq = np.array(y_seq)

# LSTM model
lstm_model = Sequential([
    Input(shape=(10, 1)),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_seq, y_seq, epochs=50, verbose=0)
seismic_forecast = lstm_model.predict(X_seq).flatten()

# ========== 2. Geohazard Risk Classification with Random Forest ========== #
X_geo = np.random.rand(500, 5)
y_geo = np.random.randint(0, 2, 500)  # 0: Low Risk, 1: High Risk
X_train_geo, X_test_geo, y_train_geo, y_test_geo = train_test_split(X_geo, y_geo, test_size=0.2)

geo_model = RandomForestClassifier(n_estimators=100)
geo_model.fit(X_train_geo, y_train_geo)
geo_pred = geo_model.predict(X_test_geo)

# PCA for 2D scatter visualization
pca = PCA(n_components=2)
X_geo_2D = pca.fit_transform(X_test_geo)

# ========== 3. Geothermal Site Hazard Classification with XGBoost ========== #
X_site = np.random.rand(500, 6)
y_site = np.random.choice([0, 1, 2], 500)
X_train_site, X_test_site, y_train_site, y_test_site = train_test_split(X_site, y_site, test_size=0.2)

site_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
site_model.fit(X_train_site, y_train_site)
site_pred = site_model.predict(X_test_site)

# Confusion Matrix
cm = confusion_matrix(y_test_site, site_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Moderate", "High"])

# ========== 4. Hazard Impact Score Prediction ========== #
X_impact = np.random.rand(300, 4)
y_impact = X_impact @ np.array([0.3, 0.5, 0.1, 0.2]) + np.random.rand(300) * 0.2
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_impact, y_impact, test_size=0.2)

impact_model = Sequential([
    Input(shape=(4,)),
    Dense(32, activation='relu'),
    Dense(1)
])
impact_model.compile(optimizer='adam', loss='mse')
impact_model.fit(X_train_imp, y_train_imp, epochs=50, verbose=0)
impact_pred = impact_model.predict(X_test_imp).flatten()

# ========== 4-in-1 Visualization ========== #
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot 1: Seismic Activity Forecast
axs[0, 0].plot(range(10, time_steps), y_seq, label="Actual", color='blue')
axs[0, 0].plot(range(10, time_steps), seismic_forecast, label="Forecast", color='red')
axs[0, 0].set_title("Seismic Activity Forecast (LSTM)")
axs[0, 0].legend()

# Plot 2: Geohazard Risk Scatter (PCA + Random Forest)
colors = ['green' if label == 0 else 'red' for label in geo_pred]
axs[0, 1].scatter(X_geo_2D[:, 0], X_geo_2D[:, 1], c=colors, alpha=0.6)
axs[0, 1].set_title("Geohazard Risk Classification (Random Forest)")
axs[0, 1].set_xlabel("PCA Component 1")
axs[0, 1].set_ylabel("PCA Component 2")

# Plot 3: Confusion Matrix (XGBoost)
cm_display.plot(ax=axs[1, 0], cmap='Blues', values_format='d')
axs[1, 0].set_title("Geothermal Site Hazard Level (XGBoost)")

# Plot 4: Hazard Impact Score Prediction
axs[1, 1].hist(impact_pred, bins=20, color='purple', alpha=0.7)
axs[1, 1].set_title("Predicted Hazard Impact Score")
axs[1, 1].set_xlabel("Impact Score")
axs[1, 1].set_ylabel("Frequency")

plt.suptitle("Geothermal Hazard & Risk Intelligence Dashboard", fontsize=18)
plt.show()
