import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import random

# Simulated coordinates and names of districts
districts = [
    {'name': 'Dhaka', 'lat': 23.8103, 'lon': 90.4125},
    {'name': 'Chittagong', 'lat': 22.3569, 'lon': 91.7832},
    {'name': 'Rajshahi', 'lat': 24.3745, 'lon': 88.6042},
    {'name': 'Khulna', 'lat': 22.8456, 'lon': 89.5403},
    {'name': 'Sylhet', 'lat': 24.8949, 'lon': 91.8687},
    {'name': 'Barisal', 'lat': 22.7010, 'lon': 90.3535},
    {'name': 'Rangpur', 'lat': 25.7460, 'lon': 89.2500},
    {'name': 'Mymensingh', 'lat': 24.7471, 'lon': 90.4203},
    {'name': 'Comilla', 'lat': 23.4607, 'lon': 91.1809},
    {'name': 'Jessore', 'lat': 23.1700, 'lon': 89.2000}
]

# Simulate geothermal potential score for each district
for d in districts:
    d['potential'] = round(random.uniform(0, 1), 2)

# Extract coordinates and potential for clustering and heatmap
X = np.array([[d['lat'], d['lon']] for d in districts])
potentials = np.array([d['potential'] for d in districts])

# Apply KMeans clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
labels = kmeans.labels_

# Cluster colors for legend
cluster_colors = ['blue', 'green', 'purple']
label_color_map = {i: cluster_colors[i] for i in range(k)}

# Create the plot
plt.figure(figsize=(12, 8))
plt.title("Geothermal Potential Clustering & Hotspot Map - Bangladesh", fontsize=16)

# Draw heatmap background (hotspots of geothermal potential)
sns.kdeplot(
    x=[d['lon'] for d in districts],
    y=[d['lat'] for d in districts],
    weights=potentials,
    cmap="YlOrRd",  # Yellow -> Red for heat intensity
    fill=True,
    alpha=0.5,
    bw_adjust=0.3
)

# Plot each district with color-coded clusters
for i, d in enumerate(districts):
    color = label_color_map[labels[i]]
    plt.scatter(d['lon'], d['lat'], c=color, s=150, edgecolor='black', label=f'Cluster {labels[i]}' if f'Cluster {labels[i]}' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(d['lon'] + 0.05, d['lat'] + 0.05, f"{d['name']} ({d['potential']})", fontsize=9)

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:,1], centers[:,0], c='black', s=300, marker='X', label='Cluster Center')

# Labels and legend
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.grid(True)
plt.legend(title="Legend", loc='upper right')
plt.tight_layout()
plt.show()
