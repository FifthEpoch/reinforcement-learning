import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.table import Table

rd = lambda x: round(x, 1)

class Cell:

    def __init__(self, _x, _y, _value=0.0):

        self.coord = np.array([_x, _y])
        self.v = _value
        self.count = 0


    def get_count(self):
        return self.count


    def get_v(self):
        return self.v


    def increment_count(self):
        self.count += 1


    def set_v(self, _v):
        self.v = _v


class Grid:

    def __init__(self, _w=4, _h=4, _gamma=1.0):

        self.gamma = _gamma

        self.w = _w
        self.h = _h

        self.actions = [np.array([0, -1]), np.array([0, 1]), np.array([-1, 0]), np.array([1, 0])]
        self.pi = 1 / len(self.actions)

        self.terminal_1, self.terminal_2 = np.array([0, 0]), np.array([3, 3])

        self.grid = np.array([[Cell(i, j) for j in range(_h)] for i in range(_w)])
        self.state = np.array([math.floor(_w / 2), math.floor(_h / 2)])


    def clear_data(self):
        self.grid = np.array(
            [[Cell(i, j) for j in range(self.h)] for i in range(self.w)]
        )


    def compute_V(self, _theta=0.01):

        delta = _theta * 2
        while delta > _theta:

            # refresh data memory
            data = self.copy_cell_v()

            # reset variables
            delta = 0

            # looping through all cells on grid
            for (i, j), cell in np.ndenumerate(self.grid):
                v = 0
                old_v = cell.get_v()
                state = np.array([i, j])

                # looping through all available actions
                for action in self.actions:

                    next_state, r, is_terminal = self.get_next_state_n_reward(state, action)
                    x, y = next_state[0], next_state[1]
                    v_sp = data[x, y]
                    v += self.compute_partial_v(r, v_sp)

                # completing the state-value function
                v = self.pi * v
                cell.set_v(v)

                # finding and storing max(delta)
                delta_ij = v - old_v
                delta = delta_ij if delta_ij > delta else delta

            print(f"delta = {delta}")

        self.show_grid()


    def compute_partial_v(self, _r, _v_sp):
        return _r + (self.gamma * _v_sp)


    def copy_cell_v(self):
        data = np.array([[0.0] * self.h for i in range(self.w)])
        for (i, j), cell in np.ndenumerate(self.grid):
            data[i, j] = cell.get_v()
        return data

    def get_next_state_n_reward(self, _state, _action):
        if np.all(_state == self.terminal_1) \
                or np.all(_state == self.terminal_2):
            return _state, 0, True
        else:
            next_state = np.add(_state, _action)
            if self.is_in_bounds(next_state):
                return next_state, -1, False
            else:
                return _state, -1, False


    def get_rand_action(self):
        rand = np.random.randint(0, 4)
        return self.actions[rand]


    def is_in_bounds(self, _state):
        x, y = _state[0], _state[1]
        if x < 0 or x >= self.h:
            return False
        elif y < 0 or y >= self.w:
            return False
        return True




    def monte_carlo(self, _theta=0.01, _steps=100):

        total_r = 0
        delta = _theta * 2
        while delta > _theta:

            data = self.copy_cell_v()

            for i in range(_steps):

                # get next state and reward
                next_state, r, is_terminal = self.get_next_state_n_reward(
                    _state=self.state,
                    _action=self.get_rand_action()
                )

                total_r += r

                # simulate next state's next state and reward
                next_next_state, next_r, is_terminal = \
                    self.get_next_state_n_reward(
                        _state=next_state,
                        _action=self.get_rand_action()
                    )

                x, y = self.state[0], self.state[1]
                u, v = next_state[0], next_state[1]
                f, g = next_next_state[0], next_next_state[1]

                # increment counts for cell[x, y]
                self.grid[x, y].increment_count()

                # get value estimate of next state
                partial_v_uv = self.compute_partial_v(
                    _r=next_r,
                    _v_sp=self.grid[f, g].get_v()
                )
                v_uv = self.update_mean(
                    _mean=self.grid[u, v].get_v(),
                    _val=partial_v_uv,
                    _cnt=self.grid[u, v].get_count() + 1
                )

                # update value estimate of current state
                partial_v_xy = self.compute_partial_v(r, v_uv)
                v_xy = self.update_mean(
                    _mean=self.grid[x, y].get_v(),
                    _val=partial_v_xy,
                    _cnt=self.grid[x, y].get_count()
                )
                self.grid[x, y].set_v(v_xy)

                # move to next state
                self.state = next_state

            # find max(delta) after N steps
            new_data = self.copy_cell_v()
            data = np.subtract(new_data, data)
            data = np.absolute(data)
            delta = np.amax(data)

        self.show_grid()


    def show_grid(self):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        table = Table(ax, bbox=[0, 0, 1, 1])

        for (i, j), cell in np.ndenumerate(self.grid):
            table.add_cell(
                i, j, 0.2, 0.2, text=str(rd(cell.get_v())), loc='center'
            )
        ax.add_table(table)
        plt.show()


    def update_mean(self, _mean, _val, _cnt):
        if _cnt == 0:
            return 0
        return _mean + ((_val - _mean) / _cnt)


if __name__ == '__main__':

    # general parameters
    width = 5
    height = 5
    gamma = 0.9
    theta = 0.00001

    # Monto Carlo parameters
    steps = 200

    # init Grid object
    gridworld = Grid(width, height, gamma)

    # getting the true value of each cell
    gridworld.compute_V(theta)

    # reinitialize all cells
    gridworld.clear_data()

    # monte carlo method
    gridworld.monte_carlo(theta, steps)

