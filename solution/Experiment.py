import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt
import pandas as pd

T = 90
resolution = 0.01


class Experiment:

    def __init__(self, num_agents, endpoint, see_endpoint_fire = 0., un_aware_agents=0., elder_ratio=0., agents_positions=None, room_size=15, type="normal"):
        self.agents = np.ndarray(num_agents, dtype=np.object)
        self.k = 0
        self.endpoint = endpoint
        self.room_size = room_size
        number_of_elders = round(num_agents * elder_ratio)
        number_of_unaware_agents = round(num_agents * un_aware_agents)
        endpoint_fire_agents = round(num_agents * see_endpoint_fire)
        if number_of_unaware_agents > 0 and number_of_elders > 0:
            number_of_elders = 0
        if endpoint_fire_agents > 0:
            number_of_elders = 0
            number_of_unaware_agents = 0
        print('elders: ', number_of_elders)
        print('unaware: ',number_of_unaware_agents)
        print('fire alerted: ',endpoint_fire_agents)
        if agents_positions is not None and len(agents_positions) == num_agents:
            for i in range(num_agents):
                position = agents_positions[i]
                if i < number_of_elders:
                    agent = Agent(id=i, room_size=room_size, type='elderly', x=position[0], y=position[1], endpoint=endpoint)
                elif i < number_of_unaware_agents:
                    agent = Agent(id=i, room_size=room_size, see_endpoint=False, x=position[0], y=position[1], endpoint=endpoint, type=type)
                elif i < endpoint_fire_agents:
                    agent = Agent(id=i, room_size=room_size, fire_alerted=True, x=position[0], y=position[1], endpoint=endpoint, type=type)
                else:
                    agent = Agent(id=i, room_size=room_size, x=position[0], y=position[1], endpoint=endpoint, type=type)
                self.agents[i] = agent
        else:
            for i in range(num_agents):
                cords = self.get_unique_cords()
                if i < number_of_elders:
                    agent = Agent(id=i, room_size=room_size, type='elderly', x=cords[0], y=cords[1], endpoint=endpoint)
                elif i < number_of_unaware_agents:
                    agent = Agent(id=i, room_size=room_size, see_endpoint=False, x=cords[0], y=cords[1], endpoint=endpoint, type=type)
                elif i < endpoint_fire_agents:
                    agent = Agent(id=i, room_size=room_size, fire_alerted=True, x=cords[0], y=cords[1], endpoint=endpoint, type=type)
                else:
                    agent = Agent(id=i, room_size=room_size, x=cords[0], y=cords[1], endpoint=endpoint, type=type)
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
            if self.k % 100 == 0:
                print('Current Time: {}s'.format(self.k * resolution))
            for agent in self.agents:
                if not agent.is_escaped():
                    agent.step(self.agents, self.k, resolution)
            all_escaped = self.is_all_escaped()
            self.k += 1
        if all_escaped:
            print('all agents escaped in time\nTime: {}s'.format((self.k - 1) * 0.01))
        else:
            index = 0
            for agent in self.agents:
                if agent.is_escaped():
                    index += 1
            print('escaped: ',index)

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
                    goal = agent.get_goal()
                    plt.scatter(x=goal[0], y=goal[1], color='green')
                    plt.text(goal[0], goal[1], 'End\n')
                    plt.scatter(x=x[0], y=y[0], color='red')
                    plt.text(x[0], y[0], 'Start\n')
                    plt.xlim(0, self.room_size)
                    plt.ylim(0, self.room_size)
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
    counter = 0
    for i in range(200):
        ids.append(i)
        exp = Experiment(num_agents=1, room_size=17)
        exp.run()
        agent = list(exp.agents[0].get_points().values())
        agents[i] = agent
        max_length = max(max_length, len(agent))
    times = []
    for i in range(max_length):
        values = []
        for j in range(len(agents)):
            if i < len(agents[j]):
                values.append(agents[j][i])
        times.append(values)
    index = 0
    for time in times:
        print(len(times) - index)
        index += 1
        for i in range(len(time)):
            x = np.array(time[i])
            for j in range(len(time)):
                y = np.array(time[j])
                if i == j:
                    continue
                if np.allclose(x, y, atol=0.5):
                    counter += 1
        print('collisions: ', counter)

def q_2_seif_a():
    exp = Experiment(endpoint=[[17, 17/2]],num_agents=20, room_size=17, type="random_speed")
    exp.run()


if __name__ == '__main__':
    q_2_seif_a()
    '''
    endpoints = [[17, 17/2]]
    exp = Experiment(endpoint=endpoints, see_endpoint_fire=0.5, num_agents=100, room_size=17)
    exp.run()
    '''