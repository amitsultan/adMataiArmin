import numpy as np

class Room:

    def __init__(self, room_size):
        self.size = room_size
        self.room = np.zeros(room_size ** 2).reshape(room_size, room_size)
        self.room[0, :] = np.ones(room_size)
        self.room[room_size - 1, :] = np.ones(room_size)
        self.room[:, 0] = np.ones(room_size)
        self.room[:, room_size - 1] = np.ones(room_size)
        self.door = int(room_size / 2)
        self.room[self.door, 0] = 0
        # print(self.room)

    def get_destination(self):
        return np.array([[-0.5, self.size/2]])


    def put_single_victim(self, victim):
        self.room[victim.x][victim.y] = 3
        # print(self.room)

    def move_single_victim(self, N_steps, victim):

        dt = 0.01
        tmp = 0
        agents_escaped = np.zeros(N_steps)
        num_agents = 1
        num_unkown = 2
        y = np.zeros((num_unkown, num_agents, N_steps))
        v = np.zeros((num_unkown, num_agents, N_steps))
        a = np.zeros((num_unkown, num_agents, N_steps))

        y[:, :, 0] = 6 # was originally y0 **tomer changed**
        # v[:,:,0] = v0

        for k in range(N_steps - 1):
            # print(100*k/N_steps, '% done.')
            # a[:, :, k] = f(y[:, :, k], v[:, :, k])
            v[:, :, k + 1] = v[:, :, k] + dt * a[:, :, k]
            y[:, :, k + 1] = y[:, :, k] + dt * v[:, :, k + 1]

            for i in range(num_agents):
                # checks if there are two destination and calculates the distance to the closets destination
                destination = np.zeros(len(self.get_destination()))
                for count, des in enumerate(self.get_destination()):
                    destination[count] = np.linalg.norm(y[:, i, k + 1] - des)
                distance = np.amin(destination)

                if distance < 0.1:
                    # we have to use position of door here instead of (0,5)
                    y[:, i, k + 1] = 10 ** 6 * np.random.rand(2)
                    # as well we have to change the  to some c*radii
                    tmp += 1

            agents_escaped[k + 1] = tmp
        print(y)
        return y, agents_escaped, a


if __name__ == '__main__':
    room = Room(room_size=15)

