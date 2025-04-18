import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simulated subsurface data (depth/distance)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.exp(-0.1 * X) * np.sin(2 * np.pi * X / 10)  # simulate decaying heat wave

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
y_tensor = torch.tensor(y, dtype=torch.float32).squeeze()

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PINN()

# Define physics-informed loss function
def loss_fn(model, x, y_true):
    y_pred = model(x)
    grad_y = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    # Physics loss: dH/dx + 0.1*H = 0 (simulating quantum heat decay)
    physics_loss = torch.mean((grad_y + 0.1 * y_pred) ** 2)
    data_loss = torch.mean((y_pred - y_true.unsqueeze(1)) ** 2)
    return data_loss + physics_loss

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    loss = loss_fn(model, X_tensor, y_tensor)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_tensor).detach().numpy()

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(X, y, label="True Heat Flow", linestyle='--', color='black')
plt.plot(X, predictions, label="Predicted by PINN", color='red')
plt.xlabel("Distance/Depth (km)")
plt.ylabel("Heat Flow (W/m² or scaled unit)")
plt.title("Subsurface Heat Flow Modeling using PINN")
plt.legend()
plt.grid(True)
plt.show()
