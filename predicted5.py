# Geothermal Drilling Success & Site Suitability Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Simulating Data ---
# Number of drilling sites
n_samples = 1000

# Simulated features
np.random.seed(42)
temperature = np.random.uniform(100, 250, n_samples)  # Temperature in Celsius
pressure = np.random.uniform(100, 500, n_samples)  # Pressure in bar
seismic_activity = np.random.uniform(0, 10, n_samples)  # Seismic activity index

# Simulated labels: 1 for high potential, 0 for low potential
labels = (temperature > 150) & (pressure > 300) & (seismic_activity < 3)  # High potential if high temp, pressure, and low seismic activity
labels = labels.astype(int)

# Create a DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'SeismicActivity': seismic_activity,
    'Label': labels
})

# --- Data Preprocessing ---
# Split data into features and target
X = df[['Temperature', 'Pressure', 'SeismicActivity']]
y = df['Label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- Model Training (Random Forest) ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Predictions ---
y_pred = rf_model.predict(X_test)

# --- Evaluation ---
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

# --- Feature Importance ---
feature_importances = rf_model.feature_importances_

# Plot feature importance
features = ['Temperature', 'Pressure', 'SeismicActivity']
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Geothermal Drilling Site Suitability')
plt.show()

# --- Visualizing Decision Boundaries (Optional) ---
# Create a grid of points to visualize decision boundary
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                     np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))

# Predict class labels for the grid points
Z = rf_model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.xlabel('Temperature (scaled)')
plt.ylabel('Pressure (scaled)')
plt.title('Geothermal Drilling Site Suitability Prediction (Decision Boundary)')
plt.show()


