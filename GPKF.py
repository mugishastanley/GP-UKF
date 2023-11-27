# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:38:27 2023

@author: Starde
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

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
        sigma = cholesky((n + lambda_) * self.state_covariance, lower=True)

        points[0] = self.state
        for i in range(n):
            points[i + 1] = self.state + sigma[i]
            points[i + n + 1] = self.state - sigma[i]

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

# Generate synthetic data
np.random.seed(42)
X_true = np.linspace(0, 10, 100)
y_true = np.sin(X_true)

# Add noise to the true function to simulate observed data
noise_level = 0.1
y_observed = y_true + noise_level * np.random.randn(len(X_true))

# Run the GP-UKF
initial_state = np.zeros(1)
process_noise = 0.1 * np.eye(1)
measurement_noise = 0.1 * np.eye(1)

alpha = 1.0  # You can adjust alpha and beta as needed
beta = 2.0

gp_ukf = GaussianProcessUKF(initial_state, process_noise, measurement_noise, alpha=alpha, beta=beta)


# Store the predicted states
num_steps_ahead = 5
predicted_states = np.zeros((len(X_true), num_steps_ahead))

for i in range(len(X_true)):
    # Prediction step
    gp_ukf.predict()

    # Update step with observed data
    gp_ukf.update(y_observed[i])

    # Perform additional prediction steps for multi-step ahead
    for step in range(num_steps_ahead):
        gp_ukf.predict()
        predicted_states[i, step] = gp_ukf.state.copy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_true, y_true, label="True Function", linestyle="--")
plt.plot(X_true, y_observed, label="Observed Data", marker="o", linestyle="None")

# Plot the trajectory of predictions over the horizon
for step in range(num_steps_ahead):
    plt.plot(X_true[step:len(X_true)-num_steps_ahead+step+1], predicted_states[:len(X_true)-num_steps_ahead+1, step],
             label=f"GP-UKF Prediction (Step {step + 1})", linestyle="-")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()