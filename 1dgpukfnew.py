# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:19:17 2023

@author: Starde
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class GaussianProcessUKF:
    def __init__(self, initial_state, process_noise, measurement_noise, alpha=1.0, beta=2.0, length_scale=1.0):
        self.state = initial_state
        self.state_covariance = np.eye(len(initial_state))
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.alpha = alpha
        self.beta = beta
        self.length_scale = length_scale  # Length scale for the RBF kernel
        self.points, self.weights = self.calculate_sigma_points()
        self.gp_model = GaussianProcessRegressor(kernel=RBF(length_scale=length_scale) + WhiteKernel(noise_level=process_noise))

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
            points[i + 1] = self.state + sigma[i]
            points[i + n + 1] = self.state - sigma[i]

        return points, weights

    def predict(self, steps_ahead):
        self.points, _ = self.calculate_sigma_points()

        predictions = []
        for _ in range(steps_ahead):
            # Gaussian Process Prediction with RBF kernel
            X_train = np.array(self.points)
            y_train = np.array([self.state] * len(self.points))

            # Training the GP model
            self.gp_model.fit(X_train, y_train)

            # Predicting the next state
            next_state, _ = self.gp_model.predict([self.state], return_cov=True)
            self.state = next_state[0]
            self.state_covariance = self.process_noise + _[0]

            predictions.append(self.state.copy())

        return predictions

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

# Run the GP-UKF with Gaussian Process Prediction several steps ahead
initial_state = np.zeros(1)
process_noise = 0.1 * np.eye(1)
measurement_noise = 0.1 * np.eye(1)

alpha = 1.0
beta = 2.0
length_scale = 1.0

gp_ukf = GaussianProcessUKF(initial_state, process_noise, measurement_noise, alpha=alpha, beta=beta, length_scale=length_scale)

# Store the predicted states
predicted_states = []

# Number of steps ahead to predict
steps_ahead = 5

for i in range(len(X_true)):
    # Predict several steps ahead with Gaussian Process
    predictions = gp_ukf.predict(steps_ahead)

    # Update step with observed data
    gp_ukf.update(y_observed[i])

    # Save the predicted states
    predicted_states.extend(predictions)

# Plot the results
# Plot the results
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_true, y_true, label="True Function", linestyle="--")
plt.plot(X_true, y_observed, label="Observed Data", marker="o", linestyle="None")

# Plot the predicted states for the corresponding range of X_true
predicted_steps = len(predicted_states) // steps_ahead

for i in range(steps_ahead):
    start_idx = i * predicted_steps
    end_idx = (i + 1) * predicted_steps
    plt.plot(X_true[:predicted_steps], predicted_states[start_idx:end_idx], label=f"Step {i + 1} Prediction", linestyle="-")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

