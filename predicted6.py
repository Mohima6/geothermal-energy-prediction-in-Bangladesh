import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lifelines import KaplanMeierFitter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

# ========= Generate Synthetic Data ========= #
np.random.seed(42)
num_samples = 1000
X = np.random.rand(num_samples, 5)  # Features (temp, depth, pressure, etc.)
y_efficiency = np.random.rand(num_samples) * 100  # Efficiency %
y_recovery = np.random.rand(num_samples) * 0.5 + 0.3  # Recovery [0.3–0.8]
y_success = np.random.randint(0, 2, size=num_samples)  # Binary: success/failure
y_heat_transfer = np.random.rand(num_samples) * 200 + 100  # W/m²
y_lifetime = np.random.exponential(scale=15, size=num_samples)  # Years

# ========= Split All Targets ========= #
X_train, X_test, y_success_train, y_success_test = train_test_split(X, y_success, test_size=0.2, random_state=42)
_, _, y_eff_train, y_eff_test = train_test_split(X, y_efficiency, test_size=0.2, random_state=42)
_, _, y_rec_train, y_rec_test = train_test_split(X, y_recovery, test_size=0.2, random_state=42)
_, _, y_heat_train, y_heat_test = train_test_split(X, y_heat_transfer, test_size=0.2, random_state=42)

# ========= 1. Drilling Success Prediction ========= #
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_success_train)
success_pred = clf.predict(X_test)

# ========= 2. Efficiency & Recovery Prediction ========= #
eff_model = GradientBoostingRegressor()
eff_model.fit(X_train, y_eff_train)
eff_pred = eff_model.predict(X_test)

rec_model = SVR()
rec_model.fit(X_train, y_rec_train)
rec_pred = rec_model.predict(X_test)

# ========= 3. Heat Transfer Rate Prediction ========= #
model_pinn = Sequential([
    Input(shape=(5,)),
    Dense(64, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(1)
])
model_pinn.compile(optimizer='adam', loss='mse')
model_pinn.fit(X_train, y_heat_train, epochs=50, batch_size=32, verbose=0)
heat_pred = model_pinn.predict(X_test).flatten()

# ========= 4. Reservoir Lifetime (Survival Analysis) ========= #
kmf = KaplanMeierFitter()
durations = y_lifetime
event_observed = np.random.randint(0, 2, size=num_samples)  # Random censoring

# ========= Plotting ========= #
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(hspace=0.4)

# Plot 1: Classification
axs[0, 0].scatter(range(len(y_success_test)), y_success_test, color='blue', label='Actual')
axs[0, 0].scatter(range(len(success_pred)), success_pred, color='red', alpha=0.6, label='Predicted')
axs[0, 0].set_title('Geothermal Drilling Success Prediction')
axs[0, 0].legend()

# Plot 2: Efficiency & Recovery
axs[0, 1].plot(eff_pred, label='Efficiency (%)', color='green')
axs[0, 1].plot(rec_pred, label='Recovery Factor', color='orange')
axs[0, 1].set_title('Efficiency & Recovery Prediction')
axs[0, 1].legend()

# Plot 3: Heat Transfer Rate
axs[1, 0].plot(heat_pred, label='Predicted Heat Rate (W/m²)', color='purple')
axs[1, 0].set_title('Heat Transfer Rate Prediction')
axs[1, 0].legend()

# Plot 4: Reservoir Lifetime
kmf.fit(durations, event_observed)
kmf.plot(ax=axs[1, 1], ci_show=True, color='brown')
axs[1, 1].set_title('Reservoir Lifetime Forecast')
axs[1, 1].set_ylabel('Survival Probability')

plt.suptitle('Geothermal Resource Intelligence Dashboard', fontsize=16)
plt.show()
