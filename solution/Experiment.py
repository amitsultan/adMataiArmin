import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt
import pandas as pd

T = 90
resolution = 0.01


class Experiment:

    def __init__(self, num_agents, agents_positions=None, room_size=15, endpoint=np.array([17, 17/2])):
        self.agents = np.ndarray(num_agents, dtype=np.object)
        self.k = 0
        self.endpoint = endpoint
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
#            print('Current Time: {}s'.format(self.k * resolution))
            for agent in self.agents:
                if not agent.is_escaped():
                    agent.step(self.agents, self.k, resolution)
            all_escaped = self.is_all_escaped()
            self.k += 1
#        if all_escaped:
#            print('all agents escaped in time\nTime: {}s'.format((self.k - 1) * 0.01))

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

    def plot_agent_movement(self, index=None):
        if not index:
            for agent in self.agents:
                if agent.is_escaped():
                    points = agent.get_points()
                    lists = sorted(points.items())
                    time, cords = zip(*lists)
                    x, y = zip(*cords)
                    plt.scatter(x, y)
                    plt.ylabel('Y')
                    plt.xlabel('X')
                    plt.title('Agent path')
                    plt.scatter(x=self.endpoint[0], y=self.endpoint[1], color='green')
                    plt.text(self.endpoint[0], self.endpoint[1], 'End\n')
                    plt.scatter(x=x[0], y=y[0], color='red')
                    plt.text(x[0], y[0], 'Start\n')
                    plt.xlim(0, self.room_size - 2)
                    plt.ylim(0, self.room_size - 2)
                    plt.show()

    def agents_escape_times(self):
        if self.is_all_escaped():
            times = {}
            for agent in self.agents:
                times[agent.get_id()] = round(len(agent.get_points()) * resolution, 2)
            return times
        else:
            return None


def seif_b():
    ids = []
    total = []
    for i in range(200):
        ids.append(i)
        exp = Experiment(num_agents=1, room_size=17)
        exp.run()
        times = exp.agents_escape_times()
        if times is not None:
            for id in times.keys():
                total.append(times[id])

    print('Summary of 200 random agents:')
    print('mean: ', np.mean(total))
    print('median: ', np.median(total))
    print('max: ', np.max(total))
    print('min: ', np.min(total))
    print(len(ids))
    print(len(total))
    from scipy import stats
    D, p = stats.kstest(total, "norm")
    print(D)
    print(p)

from collections import Counter
def seif_gimel():
    ids = []
    total = []
    agents = {}
    max_length = 0
    for i in range(2):
        ids.append(i)
        exp = Experiment(num_agents=1, room_size=17)
        exp.run()
        agent = list(exp.agents[0].get_points().values())
        agents[i] = agent
        max_length = max(max_length, len(agent))
    for id in agents.keys():
        while len(agents[id]) < max_length:
            agents[id].append(np.nan)
    df = pd.DataFrame(data=agents)
    for index, row in df.iterrows():
        values = np.array(list(row.values))
        size_without_nan = len(~pd.isnull(values))
        values = values[~pd.isnull(values)]
        print(values)
    #print(df)




if __name__ == '__main__':
    seif_gimel()