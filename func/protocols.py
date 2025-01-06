from .nodes import Node

class Protocol:
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
