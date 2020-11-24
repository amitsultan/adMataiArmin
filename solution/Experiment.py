import numpy as np
from Agent import Agent

T = 90
resolution = 0.01


class Experiment:

    def __init__(self, num_agents, agents_positions=None, room_size=15, endpoint=np.array([1., 8.])):
        self.agents = np.ndarray(num_agents, dtype=np.object)
        self.k = 0
        if agents_positions is not None and len(agents_positions) == num_agents:
            for i in range(num_agents):
                position = agents_positions[i]
                agent = Agent(room_size=room_size, x=position[0], y=position[1], endpoint=endpoint)
                self.agents[i] = agent
        else:
            for i in range(num_agents):
                agent = Agent(room_size=room_size, endpoint=endpoint)
                self.agents[i] = agent

    def step(self):
        all_escaped = self.is_all_escaped()
        while self.k * resolution < T and not all_escaped:
            for agent in self.agents:
                agent.step(self.agents, self.k)
            all_escaped = self.is_all_escaped()
            self.k += 1
        if all_escaped:
            print('all agents escaped in time\nTime: {}s'.format((self.k - 1) * 0.01))

    def is_all_escaped(self):
        escaped = True
        for agent in self.agents:
            if not agent.is_escaped():
                escaped = False
                break
        return escaped

if __name__ == '__main__':
    positions = [[7, 7]]
    exp = Experiment(num_agents=1, agents_positions=positions)
    exp.step()