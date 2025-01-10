import random
import numpy as np

ROUND_TRIP_TIME = 2.0

class Node:
    NextID = 0
    MsgsSent = 0
    LastSuccess = -1
    Delay = 0

    def __init__(self, env, p, true_value_func, measure_noise_var):
        self.env = env
        self.MyID = Node.NextID
        Node.NextID += 1
        self.P = p
        self.transmitting = False
        self.last_generated_time = None
        self.current_data = None
        self.true_value_func = true_value_func
        self.measure_noise_var = measure_noise_var

    def run(self):
        while True:
            #print(f'Begin/next loop of Node: {self.env.now}')
            yield self.env.timeout(1.0)
            self.transmitting = False
            #print(f'Node After timeout: {self.env.now}')
            if random.random() < self.P:
                true_values = self.true_value_func(self.env.now)
                self.current_data = np.random.multivariate_normal(true_values.copy(), self.measure_noise_var)
                self.last_generated_time = self.env.now
                #print(f'Node {self.MyID} has msg and sending at: {self.env.now}')

                # Waiting for sending and receiving the acknowledgement
                yield self.env.timeout(ROUND_TRIP_TIME)
                # Set the transmitting - It will be done immediately
                self.transmitting = True

            else:
                self.transmitting = False

class NodeRe:
    NextID = 0
    MsgsSent = 0
    LastSuccess = -1
    Delay = 0

    def __init__(self, env, p, true_value_func, measure_noise_var):
        self.env = env
        self.MyID = NodeRe.NextID
        NodeRe.NextID += 1
        self.P = p
        self.transmitting = False
        self.success = False
        self.last_generated_time = None
        self.current_data = None
        self.true_value_func = true_value_func
        self.measure_noise_var = measure_noise_var

    def run(self):
        while True:
            yield self.env.timeout(1.0)
            self.transmitting = False

            if random.random() < self.P:
                true_values = self.true_value_func(self.env.now)
                self.current_data = np.random.multivariate_normal(true_values.copy(), self.measure_noise_var)
                self.last_generated_time = self.env.now

                # Waiting for sending and receiving the acknowledgement
                yield self.env.timeout(ROUND_TRIP_TIME)
                # Set the transmitting - It will be done immediately
                self.transmitting = True

                #print(f'Node {self.MyID} has msg and sending at: {self.env.now}')

                # Move to next time step
                yield self.env.timeout(1.0)

                re = 0
                while NodeRe.LastSuccess != self.MyID and re < 3:
                    self.transmitting = False
                    # Wait random time/ Scheduling time
                    yield self.env.timeout(np.random.randint(2, 5))
                    #print(f'Node {self.MyID} re-transmission at: {self.env.now}')
                    # Waiting for sending and receiving the acknowledgement
                    yield self.env.timeout(ROUND_TRIP_TIME)
                    re += 1
                    # Set the transmitting - It will be done immediately
                    self.transmitting = True
                    yield self.env.timeout(1.0)

                print(f'Message from node {self.MyID} re-xmit {re} time')
                self.transmitting = False
                self.success = False
                NodeRe.LastSuccess = None

            else:
                self.transmitting = False

