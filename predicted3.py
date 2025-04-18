#geothermal activity planning; i mean how it is predicting the energy on underground;  on symptoms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Simulated geothermal surface data (for 30 locations)
np.random.seed(42)
n_samples = 30

# Features (Symptoms): temp_gradient, seismic_freq, hot_spring_presence
temp_gradient = np.random.uniform(30, 90, n_samples)  # degrees/km
seismic_freq = np.random.uniform(0.1, 3.0, n_samples)  # per year
hot_spring_presence = np.random.randint(0, 2, n_samples)  # 0 or 1

X = np.column_stack((temp_gradient, seismic_freq, hot_spring_presence))

# Simulated underground energy output (target, in MW) based on symptoms
y = (
        0.5 * temp_gradient +
        10 * seismic_freq +
        50 * hot_spring_presence +
        np.random.normal(0, 10, n_samples)  # noise
)

# 🔵 Step 1: Cluster the regions based on features
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)

# 🔴 Step 2: Train a regression model inside each cluster
cluster_models = {}
colors = ['red', 'green', 'blue']
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Geothermal Energy Prediction with Cluster-Based Regression", fontsize=14)

# 📊 Plot and fit regression model per cluster
for i in range(k):
    cluster_indices = np.where(clusters == i)[0]
    X_cluster = X[cluster_indices]
    y_cluster = y[cluster_indices]

    # Train regression model
    model = LinearRegression().fit(X_cluster, y_cluster)
    cluster_models[i] = model

    # Scatter plot of cluster
    ax.scatter(
        X_cluster[:, 0], X_cluster[:, 1], y_cluster,
        color=colors[i], label=f'Cluster {i}', s=60, edgecolor='black'
    )

    # Regression surface for visualization
    x_surf, y_surf = np.meshgrid(
        np.linspace(X_cluster[:, 0].min(), X_cluster[:, 0].max(), 10),
        np.linspace(X_cluster[:, 1].min(), X_cluster[:, 1].max(), 10)
    )
    z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel(), np.ones_like(x_surf.ravel())])
    ax.plot_surface(x_surf, y_surf, z_surf.reshape(x_surf.shape), color=colors[i], alpha=0.3)

# 🧭 Axes and labels
ax.set_xlabel('Temperature Gradient (°C/km)', fontsize=12)
ax.set_ylabel('Seismic Activity (events/year)', fontsize=12)
ax.set_zlabel('Predicted Energy Output (MW)', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()
