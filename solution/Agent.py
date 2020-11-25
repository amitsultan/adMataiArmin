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
    def __init__(self, id, room_size, endpoint, type="normal", x=None, y=None):
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
        self.V_0 = 1.5 * self.multiplier
        if x is None or y is None:
            self.x[0] = np.array([np.random.randint(low=1, high=room_size - 1), np.random.randint(low=1, high=room_size - 1)])
        else:
            self.x[0] = np.array([x, y])
        self.end = endpoint
        self.m = 80
        self.v[0] = 0
        self.escaped = False

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
        self.e[k] = self.end - self.x[k]
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


