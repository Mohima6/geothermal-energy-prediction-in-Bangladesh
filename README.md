
# 🌋 Machine Learning-Driven Geothermal Energy Exploration for Bangladesh 🇧🇩

## 🔥 Project Overview

it is a comprehensive project integrating machine learning, physics-informed modeling, clustering, classification, regression, time-series forecasting, and reinforcement learning to analyze, predict, and simulate geothermal energy potential and related geohazards specifically in Bangladesh.  

By combining real geothermal gradient data with synthetic simulations, the project supports sustainable geothermal exploration and energy planning tailored to Bangladesh's regional characteristics.


## 🧩 Project Components

### 1. ⚛️ Physics-Informed Neural Network (PINN) for Heat Flow Prediction 

- Implements a physics-informed neural network using PyTorch to model subsurface heat flow as a decaying heat wave.
- The model is trained to minimize a composite loss combining data error and physics constraints (differential equation dH/dx + 0.1 * H = 0).
- Visualizes the predicted heat flow vs. true synthetic data for validation.

### 2. 📍 Geothermal Potential Clustering and Hotspot Mapping 

- Uses KMeans clustering on geospatial coordinates (latitude, longitude) of Bangladeshi districts.
- Assigns a randomized geothermal potential score to each district to simulate geothermal activity intensity.
- Visualizes clusters on a heatmap background, highlighting hotspots of geothermal potential across Bangladesh’s regions.

### 3. 📊 Cluster-Based Regression for Energy Output Prediction 

- Generates synthetic geothermal surface features: temperature gradient, seismic frequency, hot spring presence.
- Uses KMeans clustering to group similar geothermal sites.
- Trains a separate linear regression model in each cluster to predict underground energy output (MW).
- Visualizes data clusters and regression surfaces in 3D to explain how features relate to energy prediction.

### 4. 🛠️ Geothermal Drilling Site Suitability Classification 

- Simulates geothermal drilling site data with features: temperature, pressure, seismic activity.
- Labels sites as “high potential” or “low potential” based on threshold criteria.
- Trains a Random Forest classifier to predict drilling success.
- Reports classification metrics and visualizes feature importance and decision boundaries.

### 5. ⚡ National Grid Energy Forecasting and Stability Simulation 

- Simulates multiple power grid metrics: load, power generation, energy distribution, grid stability, power fluctuations, cost optimization.
- Trains an LSTM neural network to forecast electrical load demand.
- Visualizes time-series data and forecast results, demonstrating capacity to simulate and predict grid stability and optimize energy cost.

### 6. 🌐 Multi-Model Geothermal Hazard & Risk Intelligence Dashboard 

- Implements multiple ML models:
  - LSTM for seismic activity forecasting
  - Random Forest for geohazard risk classification
  - XGBoost for geothermal site hazard level classification
  - Dense neural network for hazard impact score prediction
- Integrates PCA for visualizing geohazard classification clusters.
- Displays confusion matrix and impact score distributions to support risk-informed decision-making.

### 7. 🎮 Geothermal Simulation Environment & Stacked Regression Model 

- Generates synthetic geothermal heat flow data based on depth and thermal conductivity.
- Preprocesses and smooths geothermal data.
- Defines an OpenAI Gym environment to simulate geothermal exploration navigation with discrete actions (up, down, left, right).
- Implements a stacking regressor combining KNN, SVR, and linear regression to predict geothermal activity.
- Visualizes actual vs predicted geothermal data and simulates exploratory agent movements with reward based on proximity to geothermal hotspots.

---

## 📊 Data Description

- **Real Data:**  
  Geothermal gradient and depth data from multiple Bangladeshi regions (Sylhet, Chittagong, Dhaka, Khulna, Rajshahi, Rangpur, Comilla, Mymensingh, Jessore).

- **Synthetic Data:**  
  - Heat flow and temperature profiles generated using physical equations and noise modeling.  
  - Geospatial district data simulated for clustering geothermal potential.  
  - Surface geothermal symptoms and underground energy output synthesized for regression and classification models.  
  - Time series data for seismic activity and power grid simulation.  
  - Risk and hazard classification datasets for supervised learning.


🌟 Project Insights and Impact:

Demonstrates the value of physics-informed ML in modeling geothermal subsurface phenomena.

Enables effective regional clustering and identification of geothermal hotspots in Bangladesh.

Provides predictive tools for site suitability and geohazard risk, essential for safe geothermal drilling.

Forecasts seismic and energy grid behaviors to enhance national energy planning and stability.

Simulates exploration via a reinforcement learning environment, aiding practical navigation strategies.

Supports sustainable development of geothermal energy as a renewable source for Bangladesh.

---

## 🛠️ Installation and Requirements

### Python Environment

- Python 3.8+
- Core libraries:  
  `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `torch`, `gym`, `xgboost`

### Setup

Install required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch gym xgboost

