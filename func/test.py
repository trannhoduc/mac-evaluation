import random
import simpy
import numpy as np
import matplotlib.pyplot as plt
from systems import VehicleSystem
from protocols import *


class Node:
    """Class to represent a node in the network."""
    NextID = 0

    def __init__(self, env, p, true_value_func, measure_noise_var):
        self.env = env
        self.MyID = Node.NextID
        Node.NextID += 1
        self.P = p
        self.transmitting = False
        self.last_generated_time = None
        self.current_data = None  # Current sensor data
        self.true_value_func = true_value_func  # Function to get the true value
        self.measure_noise_var = measure_noise_var

    def run(self):
        while True:
            yield self.env.timeout(1.0)
            if random.random() < self.P:
                self.transmitting = True
                true_values = self.true_value_func(self.env.now)
                self.current_data = np.random.multivariate_normal(true_values.copy(), self.measure_noise_var)
                self.last_generated_time = self.env.now
            else:
                self.transmitting = False


class Server:
    """Class to represent the server with Kalman Filtering."""

    def __init__(self, F, H, Q, R, initial_state, true_value_func):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.state = initial_state  # Initial state estimate
        self.covariance = np.eye(F.shape[0]) * 1000  # Initial covariance estimate
        self.true_states = []  # Store true states for visualization
        self.estimated_states = []  # Store estimated states
        self.sensor_readings = []  # Store sensor readings
        self.true_value_func = true_value_func  # Function to compute true values

    def kalman_filter(self, measurements=None, time=None):
        # Update true state every step using the provided true value function
        true_state = self.true_value_func(time)
        self.true_states.append(true_state.copy())

        if measurements is not None:
            self.sensor_readings.append(measurements.copy())
            measurements = [measurements[0], measurements[1], 0, 0]
        else:
            self.sensor_readings.append([None] * self.H.shape[0])

        # Prediction step
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q

        if measurements is not None:
            # Update step
            S = self.H @ predicted_covariance @ self.H.T + self.R
            K = predicted_covariance @ self.H.T @ np.linalg.inv(S)
            self.state = predicted_state + K @ (measurements - self.H @ predicted_state)
            self.covariance = (np.eye(K.shape[0]) - K @ self.H) @ predicted_covariance
        else:
            # If no measurements, only prediction
            self.state = predicted_state
            self.covariance = predicted_covariance

        # Store the estimated state
        self.estimated_states.append(self.state.copy())

        return self.state


class Protocol:
    """Base class for communication protocols."""

    def __init__(self, env, nodes, server):
        self.env = env
        self.nodes = nodes
        self.server = server

    def run(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SlottedAloha(Protocol):
    """Implementation of the Slotted ALOHA protocol."""

    def __init__(self, env, nodes, server):
        super().__init__(env, nodes, server)
        self.AoI = [0]
        self.time_slots = 0

    def run(self):
        while True:
            yield self.env.timeout(1.0)

            transmitting_nodes = [node for node in self.nodes if node.transmitting]

            if len(transmitting_nodes) == 1:
                node_sent = transmitting_nodes[0]
                Node.MsgsSent += 1
                measurements = node_sent.current_data
                estimated_state = self.server.kalman_filter(measurements, time=self.env.now)
                print(f"Time {self.env.now}: Received {measurements}, Estimated state {estimated_state}")
                self.AoI.append(min(self.env.now - node_sent.last_generated_time, self.AoI[-1] + 1))
            else:
                # If no or multiple transmissions, predict only
                self.server.kalman_filter(measurements=None, time=self.env.now)
                self.AoI.append(self.AoI[-1] + 1)

            self.time_slots += 1


def run_simulation(protocol_class, N=10, P=0.2, MaxSimtime=100.0):
    Node.NextID = 0
    Node.MsgsSent = 0

    # def true_value_func():
    #    return np.array([25.0 + env.now * 0.1, 50.0 + env.now * 0.2])

    env = simpy.Environment()

    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # State transition matrix

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])  # Measurement matrix

    # Process noise
    q_pos_var = 0.01
    q_vel_var = 0.0001

    # Measurement noise
    r_pos_var = 100
    r_vel_var = 1

    Q = np.diag([q_pos_var] * 2 + [q_vel_var] * 2)  # Process noise covariance
    R = np.diag([r_pos_var] * 2 + [r_vel_var] * 2)  # Measurement noise covariance

    true_initial_state = [10.0, 10.0, 1.0, 1.0]  # Initial x_pos, y_pos, x_vel, y_vel
    vehicle_system = VehicleSystem(dt=1.0, initial_state=true_initial_state, var=Q)

    guess_initial_state = [0.0, 0.0, 0.0, 0.0]
    nodes = [Node(env, P, vehicle_system.get_value, R) for _ in range(N)]
    server = Server(F, H, Q, R, np.array(guess_initial_state), vehicle_system.get_value)

    protocol = protocol_class(env, nodes, server)
    env.process(protocol.run())

    for node in nodes:
        env.process(node.run())

    env.run(until=MaxSimtime)

    print(f"\nSimulation Results:")
    print(f"  Nodes: {N}")
    print(f"  Transmission Prob (P): {P}")
    print(f"  Total Msgs Sent: {Node.MsgsSent}")
    print(f"  Mean Throughput: {Node.MsgsSent / protocol.time_slots:.4f}")

    # plot_aoi_vs_time(protocol.AoI, np.arange(len(protocol.AoI)))
    plot_states(server.true_states, server.sensor_readings, server.estimated_states)


def plot_aoi_vs_time(AoI, time):
    plt.plot(time, AoI, ls='-')
    plt.xlabel('Time Slot')
    plt.ylabel('Age of Information (AoI)')
    plt.title('AoI vs. Time')
    plt.grid()
    plt.show()


def plot_states(true_states, sensor_readings, estimated_states):
    true_states = np.array(true_states)
    sensor_readings = np.array(sensor_readings, dtype=object)  # Handle None values properly
    estimated_states = np.array(estimated_states)

    time = np.arange(len(true_states))
    plt.figure(figsize=(12, 12))

    for i in range(true_states.shape[1]):
        plt.subplot(4, 1, i + 1)
        plt.plot(time, true_states[:, i], label=f"True State {i + 1}", linestyle='--')

        # Extract valid sensor readings
        valid_data = [(t, sr[i]) for t, sr in zip(time, sensor_readings) if sr is not None and sr[i] is not None]
        if valid_data and i not in (2, 3):
            valid_times, valid_sensor_readings = zip(*valid_data)
            plt.scatter(valid_times, valid_sensor_readings, label=f"Sensor Reading {i + 1}", alpha=0.6, linewidths=0.01)

        plt.plot(time, estimated_states[:, i], label=f"Estimated State {i + 1}")
        plt.xlabel('Time Slot')
        plt.ylabel(f'State Value {i + 1}')
        plt.legend()
        plt.grid()

    plt.suptitle('True States, Sensor Readings, and Kalman Filter Estimates')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    run_simulation(SlottedAloha, N=10, P=0.2, MaxSimtime=200.0)