import matplotlib.pyplot as plt
import numpy as np

def plot_aoi_vs_time(AoI, time):
    plt.plot(time, AoI, ls='-')
    plt.xlabel('Time Slot')
    plt.ylabel('Age of Information (AoI)')
    plt.title('AoI vs. Time')
    plt.grid()
    plt.show()

def plot_states(true_states, sensor_readings, estimated_states):
    true_states = np.array(true_states)
    sensor_readings = np.array(sensor_readings, dtype=object)
    estimated_states = np.array(estimated_states)

    time = np.arange(len(true_states))
    plt.figure(figsize=(12, 12))

    for i in range(true_states.shape[1]):
        plt.subplot(4, 1, i + 1)
        plt.plot(time, true_states[:, i], label=f"True State {i+1}", linestyle='--')

        valid_data = [(t, sr[i]) for t, sr in zip(time, sensor_readings) if sr is not None and sr[i] is not None]
        if valid_data and i not in (2, 3):
            valid_times, valid_sensor_readings = zip(*valid_data)
            plt.scatter(valid_times, valid_sensor_readings, label=f"Sensor Reading {i+1}", alpha=0.6)

        plt.plot(time, estimated_states[:, i], label=f"Estimated State {i+1}")
        plt.xlabel('Time Slot')
        plt.ylabel(f'State Value {i+1}')
        plt.legend()
        plt.grid()

    plt.suptitle('True States, Sensor Readings, and Kalman Filter Estimates')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
