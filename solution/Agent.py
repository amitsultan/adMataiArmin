import numpy as np
import Room as room
import numpy as np

MAX_SPEED = 1.5  # m/s
A = 2 * pow(10, 3)
B = 0.08
Z = 0.5

class Agent:

    # endpoint is the door position (x, y)
    def __init__(self, room_size, endpoint, x=None, y=None, v=1.5):
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
        self.v[0] = v
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
        #self.exit_direction(k)
        self.e[k] = self.end - self.x[k]
        direction = np.zeros(2)
        # e = (0/1/-1, 0/1/-1)
        if round(self.e[k][0], 4) != 0:
            direction[0] = self.e[k][0] / abs(self.e[k][0])
        if round(self.e[k][1], 4) != 0:
            direction[1] = self.e[k][1] / abs(self.e[k][1])
#        acceleration = 0
#        if self.v[k] < MAX_SPEED:
#            # TODO add acceleration and change speed
#            total_velocity = 2
#        self.v[k+1] = self.v[k] + acceleration
        self.x[k + 1] = self.x[k] + direction * self.v[k] * interval
        if abs(self.x[k + 1][0] - self.end[0]) < self.v[k]:
            self.x[k + 1][0] = self.end[0]
        if abs(self.x[k + 1][1] - self.end[1]) < self.v[k]:
            self.x[k + 1][1] = self.end[1]
        print('-----------------------')
        print('d: ', direction)
        print('e: ', self.e[k])
        print('x: ', self.x[k + 1])
        if np.allclose(self.x[k + 1], self.end):
            self.escaped = True

    def is_escaped(self):
        return self.escaped


if __name__ == '__main__':
    victim1 = Agent(room_size=15, endpoint=np.array([0, 7]))
    victim2 = Agent(room_size=15, endpoint=np.array([0, 7]), x=6, y=6)

