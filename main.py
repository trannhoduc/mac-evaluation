import simpy
import numpy as np
import matplotlib.pyplot as plt
from func.protocols import SlottedAloha
from func.nodes import Node, NodeRe
from func.server import Server
from func.systems import VehicleSystem
from utils.helpers import plot_aoi_vs_time, plot_states

def run_simulation(Node, protocol_class, N=10, P=0.2, MaxSimtime=100.0):
    # Simulation setup
    Node.NextID = 0
    Node.MsgsSent = 0
    Node.Delay = 0
    Node.LastSuccess = -1

    # Initial simpy environment
    env = simpy.Environment()

    # Initial parameters for Kalman Filter
    F = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    q_pos_var = 0.01
    q_vel_var = 0.0001
    r_pos_var = 100
    r_vel_var = 1

    Q = np.diag([q_pos_var] * 2 + [q_vel_var] * 2)
    R = np.diag([r_pos_var] * 2 + [r_vel_var] * 2)

    true_initial_state = [10.0, 10.0, 1.0, 1.0]

    # Create simulation system
    vehicle_system = VehicleSystem(dt=1.0, initial_state=true_initial_state, var=Q)

    # Create nodes and server for simulation
    guess_initial_state = [0.0, 0.0, 0.0, 0.0]
    nodes = [Node(env, P, vehicle_system.get_value, R) for _ in range(N)]
    server = Server(F, H, Q, R, np.array(guess_initial_state), vehicle_system.get_value)

    # Run the nodes
    for node in nodes:
        env.process(node.run())

    # Initial protocol mechanism
    protocol = protocol_class(env, Node, nodes, server)
    env.process(protocol.run())

    # Run the simulation
    env.run(until=MaxSimtime)

    # Results and plotting
    print(f"\nSimulation Results:")
    print(f"  Nodes: {N}")
    print(f"  Transmission Prob (P): {P}")
    print(f"  Total Msgs Sent: {Node.MsgsSent}")
    print(f"  Mean Delay: {Node.Delay / protocol.time_slots:.4f}")
    print(f"  Mean Throughput: {Node.MsgsSent / protocol.time_slots:.4f}")

    # Visualization
    plot_aoi_vs_time(protocol.AoI, np.arange(len(protocol.AoI)))
    plot_states(server.true_states, server.sensor_readings, server.estimated_states)

if __name__ == '__main__':
    run_simulation(Node, SlottedAloha, N=100, P=0.01, MaxSimtime=200.0)
    run_simulation(NodeRe, SlottedAloha, N=100, P=0.01, MaxSimtime=200.0)
    plt.show()
