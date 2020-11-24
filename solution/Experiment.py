import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt


T = 90
resolution = 0.01


class Experiment:

    def __init__(self, num_agents, agents_positions=None, room_size=15, endpoint=np.array([2.5, 2.5])):
        self.agents = np.ndarray(num_agents, dtype=np.object)
        self.k = 0
        self.room_size = room_size
        if agents_positions is not None and len(agents_positions) == num_agents:
            for i in range(num_agents):
                position = agents_positions[i]
                agent = Agent(id=i, room_size=room_size, x=position[0], y=position[1], endpoint=endpoint)
                self.agents[i] = agent
        else:
            for i in range(num_agents):
                cords = self.get_unique_cords()
                agent = Agent(id=i, room_size=room_size, x=cords[0], y=cords[1], endpoint=endpoint)
                self.agents[i] = agent
        print('Done __init__ Experiment')

    def get_unique_cords(self):
        while True:
            x = np.random.randint(low=1, high=self.room_size - 1)
            y = np.random.randint(low=1, high=self.room_size - 1)
            is_bad = False
            for agent in self.agents:
                if agent is None:
                    break
                start_cord = agent.get_start_cords()
                if start_cord[0] == x and start_cord[1] == y:
                    is_bad = True
                    break
            if not is_bad:
                return [x, y]

    def run(self):
        all_escaped = self.is_all_escaped()
        while self.k * resolution < T and not all_escaped:
            print('Current Time: {}s'.format(self.k * resolution))
            for agent in self.agents:
                if not agent.is_escaped():
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

    def plot_agent_v(self, index=None):
        if not index:
            for agent in self.agents:
                if agent.is_escaped():
                    velocity = agent.get_velocity()
                    lists = sorted(velocity.items())
                    x, y = zip(*lists)
                    t = []
                    v = []
                    for i in range(len(y)):
                        v.append(np.linalg.norm(y[i]))
                        t.append(x[i] / 100)
                    plt.ylabel('Velocity')
                    plt.xlabel('Seconds')
                    plt.title('Velocity graph')
                    plt.plot(t, v)
                    plt.show()

    def plot_agent_a(self, index=None):
        if not index:
            for agent in self.agents:
                if agent.is_escaped():
                    velocity = agent.get_acceleration()
                    lists = sorted(velocity.items())
                    x, y = zip(*lists)
                    a = []
                    t = []
                    for i in range(len(y)):
                        a.append(np.linalg.norm(y[i]))
                        t.append(x[i] / 100)
                    plt.plot(t, a)
                    plt.ylabel('Acceleration')
                    plt.xlabel('Seconds')
                    plt.title('Acceleration graph')
                    plt.show()

if __name__ == '__main__':
    exp = Experiment(num_agents=200, room_size=17)
    exp.run()
    exp.plot_agent_v()