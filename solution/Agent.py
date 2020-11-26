import numpy as np
import Room as room

MAX_SPEED = 1.5  # m/s
A = 2 * pow(10, 3)
B = 0.08
Z = 0.5
tao = 0.5
global_collisions = 0


class Agent:

    # endpoint is the door position (x, y)
    def __init__(self, id, room_size, endpoint, type="normal", x=None, y=None, see_endpoint=True, fire_alerted=False):
        self.ID = id
        # each array index is (k = index)
        # x[0] = (x, y)
        self.x = {}  # vector of location in 2D space
        # e[0] = end - x[0]
        self.e = {}  # vector of preferred move direction
        # v[0] = V_k
        self.v = {}
        self.a = {}  # vector of acceleration for both (x , y) directions
        if type == "normal":
            self.multiplier = 1
        elif type == "elderly":
            self.multiplier = 1 / 3
        elif type == "random_speed":
            self.multiplier = 1
            self.V_0 = np.random.uniform(0.8, 1.5)
        self.V_0 = self.V_0  * self.multiplier
        print(self.V_0)
        if x is None or y is None:
            self.x[0] = np.array([np.random.randint(low=1, high=room_size - 1), np.random.randint(low=1, high=room_size - 1)])
        else:
            self.x[0] = np.array([x, y])
        if see_endpoint:
            closest_endpoint = float("inf")
            for end in endpoint:
                if np.linalg.norm(np.array(self.x[0] - end)) < np.linalg.norm(np.array(self.x[0] - closest_endpoint)):
                    closest_endpoint = end
            self.end = np.array(closest_endpoint)
        else:
            self.end = np.array(endpoint[0])
        self.m = 80
        self.v[0] = 0
        self.escaped = False
        self.fire_alerted = fire_alerted

    def get_position(self, k):
        if k in self.x:
            return self.x[k]
        return None

#    def exit_direction(self, k):
#        self.e[k] = self.end - self.x[k]
#        if self.e[k][0] != 0:
#            self.e[k][0] = self.e[k][0] / abs(self.e[k][0])
#        if self.e[k][1] != 0:
#            self.e[k][1] = self.e[k][1] / abs(self.e[k][1])

    def step(self, agents, k, interval=0.01):
        if self.escaped:
            return True
        if not self.fire_alerted:
            self.e[k] = self.end - self.x[k]
        else:
            self.get_direction_from_agents(agents, k)
        if np.linalg.norm(self.v[k]) < MAX_SPEED:
            self.a[k] = (self.V_0 - self.v[k]) / tao
            self.v[k + 1] = (self.v[k] + interval * self.a[k])
        else:
            self.a[k] = 0
            self.v[k + 1] = self.v[k]
        direction = np.zeros(2)
        max_pull = max(abs(self.e[k][0]), abs(self.e[k][1]))
        if self.e[k][0] != 0:
            direction[0] = self.e[k][0] / max_pull
        if self.e[k][1] != 0:
            direction[1] = self.e[k][1] / max_pull
        self.x[k + 1] = self.x[k] + direction * self.v[k + 1] * interval
        self.fix_small_errors(k)
        if self.agents_collision(k, agents):
            destination = np.linalg.norm(self.x[k + 1] - self.end)
            if destination < 0.1:
                self.x[k + 1] = self.end
            if np.allclose(self.x[k + 1], self.end):
                self.escaped = True
                print('ID: {} Escaped'.format(self.ID))
                return True
        else:
            self.x[k + 1] = self.x[k]
            self.v[k + 1] = 0
            self.a[k + 1] = 0

    def get_direction_from_agents(self, agents, k):
        if np.linalg.norm(self.x[k] - self.end) <= 5:
            self.e[k] = self.end - self.x[k]
            return
        x_avg = []
        y_avg = []
        for agent in agents:
            if not agent.is_escaped() and np.linalg.norm(agent.x[k] - self.x[k]) <= 5:
                distance = agent.x[k] - self.x[k]
                x_avg.append(distance[0])
                y_avg.append(distance[1])
        if len(x_avg) == 0 and k >= 1:
            self.e[k] = self.e[k - 1]
        else:
            self.e[k] = np.array([np.mean(x_avg), np.mean(y_avg)])
        return

    def fix_small_errors(self, k):
        min_x = min(self.x[k][0], self.x[k + 1][0])
        max_x = max(self.x[k][0], self.x[k + 1][0])
        min_y = min(self.x[k][1], self.x[k + 1][1])
        max_y = max(self.x[k][1], self.x[k + 1][1])
        if min_x <= self.end[0] <= max_x:
            self.x[k + 1][0] = self.end[0]
        if min_y <= self.end[1] <= max_y:
            self.x[k + 1][1] = self.end[1]

    def agents_collision(self, k, agents):
        global global_collisions
        for i in range(self.ID):
            agent = agents[i]
            if agent.is_escaped():
                continue
            if np.allclose(self.x[k + 1], agent.x[k + 1], atol=0.5):
                global_collisions += 1
                return False
        return True

    def is_escaped(self):
        return self.escaped

    def get_goal(self):
        return self.end

    def get_velocity(self):
        return self.v

    def get_acceleration(self):
        return self.a

    def get_points(self):
        return self.x

    def get_start_cords(self):
        return self.x[0]

    def get_id(self):
        return self.ID


