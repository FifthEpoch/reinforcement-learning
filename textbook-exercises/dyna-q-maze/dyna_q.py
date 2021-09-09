import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

rd = lambda x: round(x, 1)

def update_q(_s, _a, _r, _sp, _q, _alpha, _gamma):
    
    max_q_sp = np.amax(_q[_sp[0], _sp[1]])
    q_s = _q[_s[0], _s[1], _a]

    _q[_s[0], _s[1], _a] = \
        q_s + (_alpha * (_r + (_gamma * max_q_sp) - q_s))

    return _q

def show_maze(_h, _w, _q, _visited, _runs, _episodes, _avg_t, _n, _eps, _alpha, _gamma):
    # process visits per state
    visited_s = np.array([[0 for j in range(_w)] for i in range(_h)])
    for i in range(h):
        for j in range(_w):
            visited_s[i, j] = np.sum(_visited[i, j])

    norm = plt.Normalize(np.amin(visited_s) - 100, np.amax(visited_s))
    colours = plt.cm.hot(norm(visited_s))

    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='hot'), ax=ax)

    # self.actions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
    dir = {
        0: '⬅',
        1: '➡',
        2: '⬆',
        3: '⬇',
    }
    for (i, j), cell in np.ndenumerate(env):
        is_wall = False
        for w in wall:
            if np.all(w == [i, j]):
                is_wall = True
                break
        if is_wall:
            table.add_cell(
                i, j, 0.2, 0.2, loc='center'
            )
            table[(i, j)].set_facecolor('grey')
        else:
            # get dominant action of this state
            q_ij = _q[i, j]
            max_index = np.where(q_ij == np.amax(q_ij))[0]
            arrows = ''
            for k in max_index:
                arrows += dir.get(k, '')

            table.add_cell(
                i, j, 0.2, 0.2, text=arrows, loc='center'
            )
            table[(i, j)].set_facecolor(colours[i, j])

    table.auto_set_font_size(False)
    table.scale(4, 4)
    ax.add_table(table)

    plt.suptitle(f"Dyna-Q Maze n={_n}, [avg. t={rd(_avg_t)}, "
                 f"episode/run={_episodes}, ε={_eps}, α={_alpha}, γ={_gamma}] "
                 f"averaged over {_runs} runs")
    plt.show()

# init environment --------------------------------------------------------------------
h, w = 6, 9
env = np.array([[0 for j in range(w)] for i in range(h)])
alpha = 0.1
eps = 0.1
gamma = 0.95
runs = 30
episodes = 1500
n = 5

# set start and end point
start = np.array([2, 0])
goal = np.array([0, 8])

# make walls
wall = np.array([[0, 7], [1, 7], [2, 7], [1, 2], [2, 2], [3, 2], [4, 5]])
for i in range(wall.shape[0]):
    y, x = wall[i, 0], wall[i, 1]
    env[y, x] = 1

# 4 action-values per cell of dimension h x w
avg_q = np.array([[[0.0] * 4 for j in range(w)] for i in range(h)])
avg_visited = np.array([[[0] * 4 for j in range(w)] for i in range(h)])
avg_t = 0

for run in range(runs):

    # 4 action-values per cell of dimension h x w
    a = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
    q = np.array([[[0.0] * 4 for j in range(w)] for i in range(h)])
    visited = np.array([[[0] * 4 for j in range(w)] for i in range(h)])
    model = np.array([[[{'sp': np.array([-1, -1]), 'r': -1}] * 4 for j in range(w)] for i in range(h)])

    for ep in range(episodes):

        t, s, terminal = 0, start, False

        while not terminal:

            # get coordinates and action-values
            s_y, s_x = s[0], s[1]
            q_s = q[s_y, s_x]

            # get ε-greedy action
            max_i = \
                np.random.choice(np.where(q_s == np.amax(q_s))[0], 1)[0]

            p_s = np.array([])
            for i in range(4):
                p_i = (1.0 - eps) + (eps / 4) if i == max_i else eps / 4
                p_s = np.append(p_s, p_i)
            a_i = np.random.choice(4, 1, p=p_s)[0]

            # get s'
            sp = np.add(s, a[a_i])

            # make sure it's not wall
            for i in range(wall.shape[0]):
                if np.all(wall[i] == sp):
                    sp = s

            # make sure it is inbound
            if sp[0] < 0 or sp[0] >= h or sp[1] < 0 or sp[1] >= w:
                sp = s

            # distribute reward
            r = 0

            # if goal is reached
            if np.all(sp == goal):
                r, terminal = 1, True
                visited[goal[0], goal[1], 0] += 1

            # update action-value
            q = update_q(s, a_i, r, sp, q, alpha, gamma)

            # update visited
            visited[s_y, s_x, a_i] += 1

            # storing future-state and reward pair in model
            model[s_y, s_x, 0]['sp'] = sp
            model[s_y, s_x, 0]['r'] = r

            for i in range(n):
                # get random visited state and action
                visits = np.where(visited > 0)
                rand_s_a = np.random.randint(visits[0].size)
                s_y_n = visits[0][rand_s_a]
                s_x_n = visits[1][rand_s_a]
                a_n = visits[2][rand_s_a]

                # get future-state and reward pair for the state-action pair
                sp_n = model[s_y_n, s_x_n, 0]['sp']
                r_n = model[s_y_n, s_x_n, 0]['r']

                # update action-value of random state based on model
                q = update_q(np.array([s_y_n, s_x_n]), a_n, r_n, sp_n, q, alpha, gamma)

            s = sp
            t += 1

        # print(f'EP {ep} TERMINAL STATE REACHED\nq = \n{q}')
        avg_t += (t - avg_t) / ((run * episodes)+(ep + 1))

    # update averages
    for i in range(h):
        for j in range(w):
            for a in range(4):
                avg_q[i, j, a] += (q[i, j, a] - avg_q[i, j, a])/(run + 1)
                avg_visited[i, j, a] += (visited[i, j, a] - avg_visited[i, j, a])/(run + 1)

show_maze(h, w, avg_q, avg_visited, runs, episodes, avg_t, n, eps, alpha, gamma)