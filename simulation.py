import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import gym
from gym import spaces

# 1. Data Collection: Synthetic Data Generation
def generate_synthetic_geothermal_data():
    # Simulating geothermal data
    depth = np.linspace(0, 1000, 100)  # Depth in meters (0 to 1000 meters)
    thermal_conductivity = 2.5  # W/mK (typical for the Earth's crust)
    surface_temperature = 25  # Surface temperature in Celsius
    geothermal_gradient = 30  # Geothermal gradient in Celsius per kilometer

    # Temperature profile using geothermal gradient
    temperature_profile = surface_temperature + geothermal_gradient * depth / 1000  # Linear increase with depth

    # Simulated heat flow (Fourier's Law)
    heat_flow = -thermal_conductivity * np.gradient(temperature_profile, depth)

    return heat_flow

# 2. Preprocessing: Normalization and Smoothing
def preprocess_data(data):
    # Normalize the data using Min-Max Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))

    # Handle Missing Data: Impute missing values with mean
    data_normalized[np.isnan(data_normalized)] = np.mean(data_normalized)

    # Smooth data using moving average (for noise reduction)
    window_size = 10
    smoothed_data = np.convolve(data_normalized.flatten(), np.ones(window_size)/window_size, mode='valid')

    return smoothed_data, scaler

# 3. Geothermal Navigation Environment (Using OpenAI Gym)
class GeothermalNavigationEnv(gym.Env):
    def __init__(self, heat_flow_data):
        super(GeothermalNavigationEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Define 4 possible actions (up, down, left, right)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Geothermal heat flow values
        self.heat_flow_data = heat_flow_data
        self.state = None

    def reset(self):
        # Random initial state from heat flow data
        self.state = np.random.choice(self.heat_flow_data)
        return np.array([self.state])

    def step(self, action):
        # Simulate movement through geothermal zones
        if action == 0:  # Move up
            self.state += 0.05
        elif action == 1:  # Move down
            self.state -= 0.05
        elif action == 2:  # Move left
            self.state -= 0.025
        elif action == 3:  # Move right
            self.state += 0.025

        # Reward based on how close to a hotspot (hypothetical reward structure)
        reward = -abs(self.state - 0.5)  # Example: reward is higher closer to the hotspot
        done = False  # End condition can be based on exploration bounds
        return np.array([self.state]), reward, done, {}

    def render(self):
        print(f"Geothermal Activity Level: {self.state}")

# 4. Training a Stacking Regressor Model for Geothermal Prediction
def train_geothermal_model(X, y):
    # Define base learners for stacking
    base_learners = [
        ('knn', KNeighborsRegressor(n_neighbors=3)),
        ('svr', SVR(kernel='rbf')),
    ]

    # Define the final estimator
    final_estimator = LinearRegression()

    # Create the stacking model
    stacked_model = StackingRegressor(estimators=base_learners, final_estimator=final_estimator)

    # Train the model
    stacked_model.fit(X, y)

    return stacked_model

# Main script execution
if __name__ == "__main__":
    # Step 1: Generate synthetic geothermal data
    geothermal_data = generate_synthetic_geothermal_data()

    # Step 2: Preprocess the geothermal data
    smoothed_data, scaler = preprocess_data(geothermal_data)

    # Step 3: Create the geothermal navigation environment (for simulation)
    env = GeothermalNavigationEnv(smoothed_data)

    # Step 4: Train a geothermal prediction model using stacking
    X_train, X_test, y_train, y_test = train_test_split(smoothed_data.reshape(-1, 1), smoothed_data, test_size=0.2, random_state=42)

    # Train the stacked model
    model = train_geothermal_model(X_train, y_train)

    # Step 5: Use the model for prediction
    predictions = model.predict(X_test)

    # Plot predictions vs actual data
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Geothermal Data")
    plt.plot(predictions, label="Predicted Geothermal Data", linestyle="--")
    plt.xlabel("Samples")
    plt.ylabel("Geothermal Activity (Normalized)")
    plt.title("Geothermal Data Prediction")
    plt.legend()
    plt.show()

    # Step 6: Simulate geothermal navigation in the environment (optional exploration)
    print("Simulating geothermal exploration using the navigation environment:")
    obs = env.reset()
    print(f"Initial Observation: {obs}")
    for _ in range(10):
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)
        print(f"Observation: {obs}, Reward: {reward}")
        env.render()
