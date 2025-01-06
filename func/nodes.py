import random
import numpy as np

class Node:
    NextID = 0

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
            yield self.env.timeout(1.0)
            if random.random() < self.P:
                self.transmitting = True
                true_values = self.true_value_func(self.env.now)
                self.current_data = np.random.multivariate_normal(true_values.copy(), self.measure_noise_var)
                self.last_generated_time = self.env.now
            else:
                self.transmitting = False
