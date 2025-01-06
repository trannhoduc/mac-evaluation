import numpy as np

class Server:
    def __init__(self, F, H, Q, R, initial_state, true_value_func):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.state = initial_state
        self.covariance = np.eye(F.shape[0]) * 1000
        self.true_states = []
        self.estimated_states = []
        self.sensor_readings = []
        self.true_value_func = true_value_func

    def kalman_filter(self, measurements=None, time=None):
        true_state = self.true_value_func(time)
        self.true_states.append(true_state.copy())

        if measurements is not None:
            self.sensor_readings.append(measurements.copy())
            measurements = [measurements[0], measurements[1], 0, 0]
        else:
            self.sensor_readings.append([None] * self.H.shape[0])

        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q

        if measurements is not None:
            S = self.H @ predicted_covariance @ self.H.T + self.R
            K = predicted_covariance @ self.H.T @ np.linalg.inv(S)
            self.state = predicted_state + K @ (measurements - self.H @ predicted_state)
            self.covariance = (np.eye(K.shape[0]) - K @ self.H) @ predicted_covariance
        else:
            self.state = predicted_state
            self.covariance = predicted_covariance

        self.estimated_states.append(self.state.copy())
        return self.state
