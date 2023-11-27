import matplotlib.pyplot as plt
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


# Define a simple process model
def f_process(x, dt):
   # return x + np.random.normal(0, 0.1, x.shape)
   F= np.array([1],dtype=float)
   return np.dot(F,x)

# Define a measurement function for the UKF that uses the GP mean
def h_measurement(x, gp):
    mean, _ = gp.predict(x.reshape(-1, 1), return_std=True)
    return mean

# Generate synthetic time series data (sine wave with noise)
timesteps = 25
dt = 1
x_series = np.linspace(0, 10, timesteps)
y_series = np.sin(x_series) + np.random.normal(0, 0.1, x_series.shape)

# Gaussian Process initialization
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel)
train_points = x_series[:20].reshape(-1, 1)
train_targets = y_series[:20]
gp.fit(train_points, train_targets)

# UKF initialization
Q = 0.1  # Process noise variance
points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=-0.5)
ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=dt, hx=lambda x: h_measurement(x, gp), fx=f_process, points=points)
ukf.x = np.array([0.]) #initial state
ukf.P *= 0.2 #initial uncertainity
z_std = 0.1
ukf.R = np.diag([z_std**2]) # 1 standard
ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)[0, 0]

# Now loop through the time series data and predict with the UKF
ukf_predictions = np.zeros(timesteps)
ukf_confidence = np.zeros(timesteps)
for t in range(20, timesteps):
    ukf.predict()
    ukf.update(np.array([y_series[t]]))

    ukf_predictions[t] = ukf.x[0]
    ukf_confidence[t] = ukf.P[0, 0]
    print(ukf.x, 'log-likelihood', ukf.log_likelihood)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_series, y_series, 'k-', lw=2, label='True Data')
plt.plot(x_series[20:], ukf_predictions[20:], 'r--', lw=2, label='GPUKF ahead Prediction')
plt.fill_between(x_series[20:], 
                 ukf_predictions[20:] - 2 * np.sqrt(ukf_confidence[20:]),
                 ukf_predictions[20:] + 2 * np.sqrt(ukf_confidence[20:]),
                 color='orange', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title('Multi-Step Time Series Prediction via GP-UKF')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
