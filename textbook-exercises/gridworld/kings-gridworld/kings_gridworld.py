import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

rd = lambda x: round(x, 1)

class Cell:

    def __init__(self, _y, _x):

        self.coord = np.array([_y, _x])

        self.cnt = 0
        self.q = np.array([0.0] * 8)
        self.pi = np.array([0.125] * 8)

    def cpp(self):
        self.cnt += 1

    def get_cnt(self):
        return self.cnt

    def get_pi(self):
        return self.pi

    def set_pi(self, _pi):
        self.pi = _pi

    def get_q(self):
        return self.q

    def set_q(self, _q):
        self.q = _q


class Grid:

    def __init__(self, _w, _h, _alpha, _gamma, _eps):

        self.alpha = _alpha
        self.gamma = _gamma
        self.eps = _eps

        self.w = _w
        self.h = _h

        self.grid = np.array(
            [[Cell(i, j)
              for j in range(_w)]
             for i in range(_h)]
        )
        self.wind_speed = \
            np.array([0, 0, 0, -1, -1, -1, -2, -2, -1, 0])

        # left, right, up, down
        self.num_a = 8
        self.actions = \
            np.array([[0, -1], [0, 1], [-1, 0], [1, 0],
                      [-1, 1], [1, 1], [1, -1], [-1, -1]])

        self.start = np.array([3, 0])
        self.goal = np.array([3, 7])

        self.state = self.start

    def reset_grid(self):
        self.grid = np.array(
            [[Cell(i, j) for j in range(self.w)]
                            for i in range(self.h)]
        )

    def reset_cells(self, _v=0.0):
        for index, cell in np.ndenumerate(self.grid):
            cell.set_v(_v)

    def is_in_bounds(self, _s):
        y, x = _s[0], _s[1]
        if x < 0 or x >= self.w:
            return False
        elif y < 0 or y >= self.h:
            return False
        return True

    def get_next_state_n_reward(self, _s, _a):

        sp = np.add(_s, _a)
        sp[0] += self.wind_speed[_s[1]]

        if self.is_in_bounds(sp):
            if np.all(sp == self.goal):
                return sp, 0
            else:
                return sp, -1
        else:
            sp[0] = sp[0] if sp[0] < self.h else self.h - 1
            sp[0] = sp[0] if sp[0] >= 0 else 0
            sp[1] = sp[1] if sp[1] < self.w else self.w - 1
            sp[1] = sp[1] if sp[1] >= 0 else 0

            if np.all(sp == self.goal):
                return sp, 0

            return sp, -1

    def get_soft_action(self, _pi_xy):
        action_i = np.random.choice(
            self.num_a, 1, p=_pi_xy)[0]
        return action_i

    def windy_walk(self):

        t = 0
        a = np.random.randint(0, self.num_a)
        is_T = False
        while not is_T:

            y, x = self.state[0], self.state[1]

            # get a random action from the policy's optimal action(s)
            sp, r = self.get_next_state_n_reward(
                _s=self.state,
                _a=self.actions[a]
            )

            if r == 0:
                is_T = True
                self.state = sp
                continue

            # update policy of s'
            self.update_pi(sp[0], sp[1])
            # use updated policy to get a'
            pi_sp = self.grid[sp[0], sp[1]].get_pi()
            ap = self.get_soft_action(pi_sp)

            # SARSA
            self.update_q(y, x, a, sp, ap, r)

            t += 1
            a = ap
            self.state = sp
            self.grid[y, x].cpp()

        return t


    def update_q(self, _y, _x, _a, _sp, _ap, _r):

        y_sp, x_sp = _sp[0], _sp[1]

        # getting action-value q(s, a) and
        # find maximum action-value in future state s'
        q_xy = self.grid[_y, _x].get_q()
        q_sp = self.grid[y_sp, x_sp].get_q()

        # update q value of the action taken
        q_xy[_a] = \
            q_xy[_a] + (self.alpha * (_r + (self.gamma * q_sp[_ap]) - q_xy[_a]))

        # setting q
        self.grid[_y, _x].set_q(q_xy)

    def update_pi(self, _y, _x):

        # find which action has the highest value
        # after updating action-values
        pi_xy = self.grid[_y, _x].get_pi()
        q_xy = self.grid[_y, _x].get_q()
        max_q_i = np.where(q_xy == np.amax(q_xy))[0]

        if max_q_i.size > 1:
            max_q_i = max_q_i[np.random.randint(max_q_i.size)]
        else:
            max_q_i = max_q_i[0]

        # update policy (ε-greedy)
        for i in range(self.num_a):
            if i == max_q_i:
                pi_xy[i] = 1.0 - self.eps + (self.eps / self.num_a)
            else:
                pi_xy[i] = self.eps / self.num_a

        # updating policy
        self.grid[_y, _x].set_pi(pi_xy)

    def show_pi(self, _avg_t, _episode=0, _eps=0.1, _alpha=0.5, _gamma=1.0):

        # get count data
        cnts = np.array(
            [[self.grid[i, j].get_cnt()
              for j in range(self.w)]
             for i in range(self.h)]
        )

        norm = plt.Normalize(np.amin(cnts), np.amax(cnts))
        colours = plt.cm.hot(norm(cnts))

        fig, ax = plt.subplots()
        ax.set_axis_off()
        table = Table(ax, bbox=[0, 0, 1, 1])
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='hot'), ax=ax)

        for i in range(self.h):
            for j in range(self.w):

                arrows = ''
                dir = {
                    0: '⬅',
                    1: '➡',
                    2: '⬆',
                    3: '⬇',
                    4: '⬈',
                    5: '⬊',
                    6: '⬋',
                    7: '⬉'
                }
                q_ij = self.grid[i, j].get_q()
                max_index = np.where(q_ij == np.amax(q_ij))[0]

                for k in max_index:
                    arrows += dir.get(k, '')

                table.add_cell(
                    i, j, 0.2, 0.2, text=arrows, loc='center'
                )
                table[(i, j)].set_facecolor(colours[i, j])

        table.auto_set_font_size(False)
        table.scale(4, 4)
        ax.add_table(table)

        plt.suptitle(f"King's Moves Windy Gridworld policy "
                     f"[avg. t={rd(_avg_t)}, "
                     f"episode={_episode}, "
                     f"ε={_eps}, "
                     f"α={_alpha}, "
                     f"γ={_gamma}]")
        plt.show()

    def show_grid(self):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        table = Table(ax, bbox=[0, 0, 1, 1])

        for (i, j), cell in np.ndenumerate(self.grid):
            table.add_cell(
                i, j, 0.2, 0.2,
                text=str(i) + ', ' + str(j),
                loc='center'
            )
        ax.add_table(table)
        plt.show()

if __name__ == '__main__':

    # general parameters
    width = 10
    height = 7
    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1

    episode = 300
    gridworld = Grid(width, height, alpha, gamma, epsilon)

    avg_t = 0
    for i in range(episode):
        avg_t = avg_t + ((gridworld.windy_walk() - avg_t) / (episode + 1.0))
    gridworld.show_pi(avg_t, episode, epsilon, alpha, gamma)