from numpy.random import randn
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class VehicleSystem:
    def __init__(self, dt, initial_state, var):
        self.state = np.array(initial_state)
        self.var = var
        self.dt = dt
        self.last_update = -1

    def update(self):
        # Apply state transition
        self.state[0] += self.state[2] * self.dt  # x_pos += x_vel * dt
        self.state[1] += self.state[3] * self.dt  # y_pos += y_vel * dt

        # Add process noise
        process_noise = np.random.multivariate_normal(
            np.zeros(len(self.state)),
            self.var
        )
        self.state += process_noise

    def get_value(self, time):
        # Update only once per time step
        if time > self.last_update:
            self.last_update = time
            self.update()
        return self.state

if __name__ == '__main__':
    # Process noise
    q_pos_var = 0.5
    q_vel_var = 0.2

    initial_state = [10.0, 10.0, 2.0, 2.0]  # Initial x_pos, y_pos, x_vel, y_vel

    vehicle_system = VehicleSystem(dt=1.0, initial_state=initial_state, pos_var=q_pos_var, vel_var=q_vel_var)

    states = []
    for i in range(10):
        a = vehicle_system.get_value(i)
        print(f'A is {a}')
        states.append(a.copy())

    print(states)

    states = np.array(states)  # Convert to numpy array for easier indexing
    time = np.arange(10)

    plt.figure(figsize=(12, 10))

    # Plot each feature in a separate subplot
    labels = ['x_pos', 'y_pos', 'x_vel', 'y_vel']
    for i in range(states.shape[1]):
        plt.subplot(4, 1, i + 1)
        plt.plot(time, states[:, i], label=f'{labels[i]}')
        plt.xlabel('Time')
        plt.ylabel(f'{labels[i]}')
        plt.legend()
        plt.grid()

    plt.suptitle('Vehicle System States Over Time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



