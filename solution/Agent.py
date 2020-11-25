import numpy as np
import Room as room
import numpy as np

MAX_SPEED = 1.5  # m/s
A = 2 * pow(10, 3)
B = 0.08
Z = 0.5
V_0 = 1.5
tao = 0.5


class Agent:

    # endpoint is the door position (x, y)
    def __init__(self, id, room_size, endpoint, x=None, y=None, v=1.5):
        self.ID = id
        # each array index is (k = index)
        # x[0] = (x, y)
        self.x = {}  # vector of location in 2D space
        # e[0] = end - x[0]
        self.e = {}  # vector of preferred move direction
        # v[0] = V_k
        self.v = {}
        self.a = {}  # vector of acceleration for both (x , y) directions
        if x is None or y is None:
            self.x[0] = np.array([np.random.randint(low=1, high=room_size - 1), np.random.randint(low=1, high=room_size - 1)])
        else:
            self.x[0] = np.array([x, y])
        self.end = endpoint
        self.m = 80
        self.v[0] = 0
        self.escaped = False
        print('I started at: ({}, {})'.format(self.x[0][0], self.x[0][1]))
        print('End at: ({}, {})'.format(self.end[0], self.end[1]))

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
        self.e[k] = self.end - self.x[k]
        if np.linalg.norm(self.v[k]) < MAX_SPEED:
            self.a[k] = (V_0 - self.v[k]) / tao
            self.v[k + 1] = (self.v[k] + interval * self.a[k])
        else:
            self.a[k] = 0
            self.v[k + 1] = self.v[k]
        direction = np.zeros(2)
        if self.e[k][0] != 0:
            direction[0] = self.e[k][0] / abs(self.e[k][0])
        if self.e[k][1] != 0:
            direction[1] = self.e[k][1] / abs(self.e[k][1])
        self.x[k + 1] = self.x[k] + direction * self.v[k + 1] * interval
        if self.agents_collision(k, agents):
            destination = np.linalg.norm(self.x[k + 1] - self.end)
            if destination < 0.1:
                self.x[k + 1] = self.end
            if np.allclose(self.x[k + 1], self.end):
                self.escaped = True
                print('Escaped: {}'.format(self.ID))
        else:
            self.x[k + 1] = self.x[k]
            self.v[k + 1] = 0
            self.a[k + 1] = 0

    def agents_collision(self, k, agents):
        for i in range(self.ID):
            agent = agents[i]
            if agent.is_escaped():
                continue
            if np.allclose(self.x[k + 1], agent.x[k + 1], atol=0.25):
                return False
        return True

    def is_escaped(self):
        return self.escaped

    def get_velocity(self):
        return self.v

    def get_acceleration(self):
        return self.a

    def get_start_cords(self):
        return self.x[0]



if __name__ == '__main__':
    victim1 = Agent(room_size=15, endpoint=np.array([0, 7]))
    victim2 = Agent(room_size=15, endpoint=np.array([0, 7]), x=6, y=6)