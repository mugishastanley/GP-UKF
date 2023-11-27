# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:46:26 2023

@author: Starde
"""
import numpy as np
import matplotlib.pyplot as plt
import GPy
import time

# Generate synthetic data
np.random.seed(42)
X_true = np.linspace(0, 10, 50)
y_true = np.sin(X_true)

# Add noise to the true function to simulate observed data
noise_level = 0.1
y_observed = y_true + noise_level * np.random.randn(len(X_true))

# GP model
kernel = GPy.kern.RBF(input_dim=1)
gp_model = GPy.models.GPRegression(X_true.reshape(-1, 1), y_observed.reshape(-1, 1), kernel)
gp_model.optimize_restarts(num_restarts=10)

# GPUKF implementation
class GaussianProcessUKF:
    def __init__(self, initial_state, process_noise, measurement_noise, alpha=1.0, beta=2.0):
        self.state = initial_state
        self.state_covariance = np.eye(len(initial_state))
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.alpha = alpha
        self.beta = beta
        self.points, self.weights = self.calculate_sigma_points()

    def calculate_sigma_points(self):
        n = len(self.state)
        kappa = 3 - n
        lambda_ = self.alpha**2 * (n + kappa) - n
        weights = np.full(2 * n + 1, 1.0 / (2.0 * (n + lambda_)))
        weights[0] = lambda_ / (n + lambda_)

        points = np.zeros((2 * n + 1, n))
        sigma = np.linalg.cholesky((n + lambda_) * self.state_covariance)

        points[0] = self.state
        for i in range(n):
            points[i + 1] = self.state + sigma[:, i]
            points[i + n + 1] = self.state - sigma[:, i]

        return points, weights

    def predict(self):
        self.points, _ = self.calculate_sigma_points()

        # Example prediction step - simple linear system
        self.state = np.dot(self.weights, self.points)
        self.state_covariance = self.process_noise + np.dot(
            self.weights * (self.points - self.state).T, (self.points - self.state)
        )

    def update(self, measurement):
        self.points, self.weights = self.calculate_sigma_points()

        # Example update step - simple linear system
        innovation = measurement - np.dot(self.weights, self.points)
        innovation_covariance = self.measurement_noise + np.dot(
            self.weights * (self.points - self.state).T, (self.points - self.state)
        )

        cross_covariance = np.dot(
            self.weights * (self.points - self.state).T, (self.points - self.state)
        )

        kalman_gain = cross_covariance @ np.linalg.inv(innovation_covariance)
        self.state += kalman_gain @ innovation
        self.state_covariance -= kalman_gain @ innovation_covariance @ kalman_gain.T

# Measure time for GP execution
start_time_gp = time.time()
y_pred_gp, _ = gp_model.predict(X_true.reshape(-1, 1))
end_time_gp = time.time()
execution_time_gp = end_time_gp - start_time_gp

# Measure time for GPUKF execution
start_time_ukf = time.time()
initial_state_ukf = np.zeros(1)
process_noise_ukf = 0.1 * np.eye(1)
measurement_noise_ukf = 0.1 * np.eye(1)

alpha_ukf = 1.0  # You can adjust alpha and beta as needed
beta_ukf = 2.0

gp_ukf = GaussianProcessUKF(initial_state_ukf, process_noise_ukf, measurement_noise_ukf, alpha=alpha_ukf, beta=beta_ukf)

# Store the predicted states
predicted_states_ukf = []

for i in range(len(X_true)):
    # Prediction step
    gp_ukf.predict()

    # Update step with observed data
    gp_ukf.update(y_observed[i])

    # Save the predicted state
    predicted_states_ukf.append(gp_ukf.state.copy())

end_time_ukf = time.time()
execution_time_ukf = end_time_ukf - start_time_ukf

# Calculate RMSE and MAPE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse_gp = calculate_rmse(y_true, y_pred_gp)
mape_gp = calculate_mape(y_true, y_pred_gp)

rmse_ukf = calculate_rmse(y_true, predicted_states_ukf)
mape_ukf = calculate_mape(y_true, predicted_states_ukf)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(X_true, y_true, label="True Function", linestyle="--")
plt.plot(X_true, y_observed, label="Observed Data", marker="o", linestyle="None")
plt.plot(X_true, y_pred_gp, label=f"GP Prediction (RMSE={rmse_gp:.3f}, MAPE={mape_gp:.3f}%, Time={execution_time_gp:.5f}s)", linestyle="-", linewidth=2)
plt.plot(X_true, predicted_states_ukf, label=f"GPUKF Prediction (RMSE={rmse_ukf:.3f}, MAPE={mape_ukf:.3f}%, Time={execution_time_ukf:.5f}s)", linestyle="-", linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
