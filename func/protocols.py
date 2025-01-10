#from .nodes import Node
ROUND_TRIP_TIME = 2.0

class Protocol:
    def __init__(self, env, Node, nodes, server):
        self.env = env
        self.nodes = nodes
        self.server = server
        self.Node = Node

    def run(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SlottedAloha(Protocol):
    """Implementation of the Slotted ALOHA protocol."""

    def __init__(self, env, Node, nodes, server):
        super().__init__(env, Node, nodes, server)
        self.AoI = [0]
        self.time_slots = 0
        self.Node = Node

    def run(self):
        while True:
            #print(f'Begin/next loop of Protocols: {self.env.now}')
            yield self.env.timeout(1.0)
            #print(f'Protocols after timeout: {self.env.now}')

            transmitting_nodes = [node for node in self.nodes if node.transmitting]

            if len(transmitting_nodes) == 1:
                node_sent = transmitting_nodes[0]
                self.Node.MsgsSent += 1
                self.Node.LastSuccess = node_sent.MyID
                self.Node.Delay += self.env.now - node_sent.last_generated_time
                # Normal case
                #self.AoI.append(min(self.env.now - node_sent.last_generated_time, self.AoI[-1] + 1))
                # Should be case
                self.AoI.append(min(self.env.now - node_sent.last_generated_time, self.AoI[-1] + 1))


                #print(f'Server received msg from {node_sent.MyID} at {self.env.now}')

                measurements = node_sent.current_data
                estimated_state = self.server.kalman_filter(measurements, time=self.env.now)
                #print(f"Time {self.env.now}: Received {measurements}, Estimated state {estimated_state}")

            else:
                # If no or multiple transmissions, predict only
                self.server.kalman_filter(measurements=None, time=self.env.now)
                self.AoI.append(self.AoI[-1] + 1)

            self.time_slots += 1
