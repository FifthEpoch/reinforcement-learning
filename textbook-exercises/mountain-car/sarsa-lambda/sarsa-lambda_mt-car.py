import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

runs = 10
episodes = 1000

# feature settings
h = 8
w = 8
tile_f = 8.0
num_tilings = 8
num_tilings_f = 8.0

# actions: 0, forward, reverse
actions = [0, 1, -1]
num_a = 3

# y: velocity
# x: position
total_y = 0.14
total_x = 1.7

# fundamental units: width of tile / number of tilings
y_unit = (total_y / 8.0) / tile_f
x_unit = (total_x / 8.0) / tile_f

# offseting the state variables for each tiling
offset = np.array([
    [0, 0], [-2 * y_unit, 4 * x_unit],
    [y_unit, x_unit], [-y_unit, -x_unit],
    [2 * y_unit, 2 * x_unit], [-2 * y_unit, -2 * x_unit],
    [3 * y_unit, 3 * x_unit], [-3 * y_unit, -3 * x_unit]])

# parameters
lmbd = 0.9
alphas = \
    np.array([(0.6 / num_tilings_f)])
epsilon = 0.0
gamma = 1.0

# FUNCTIONS -----------------------------------------------------------

def bound_pos(_pos):
    bd_pos = _pos if _pos <= 0.5 else 0.5
    bd_pos = bd_pos if bd_pos >= -1.2 else -1.2
    return bd_pos

def bound_vlc(_vlc):
    bd_vlc = _vlc if _vlc <= 0.07 else 0.07
    bd_vlc = bd_vlc if bd_vlc >= -0.07 else -0.07
    return bd_vlc

def est_q(_w, _fs, _a):
    q = 0.0
    for i in range(num_tilings):
        y_i, x_i = int(_fs[i * 2]), int(_fs[(i * 2) + 1])
        if x_i < 8 and y_i < 8:
            q += _w[i, y_i, x_i, _a]
    return (q / num_tilings)

def featurize(_vlc, _pos):
    # scaling to index
    f_y = math.floor((_vlc + 0.07) * (num_tilings / 0.14))
    f_x = math.floor((_pos + 1.20) * (num_tilings / 1.70))
    return int(f_y), int(f_x)

def generalize(_vlc, _pos, _asym_offset=True):
    fs = np.array([])
    for i in range(num_tilings):
        if _asym_offset:
            # asymmetric offset
            if i == 0:
                vlc_i = _vlc
                pos_i = _pos
            else:
                vlc_i = _vlc + (np.random.randint(low=-0.5, high=1.5) * y_unit)
                pos_i = _pos + (np.random.randint(low=-0.5, high=1.5) * x_unit)
        else:
            # symmetric offset
            vlc_i = _vlc + offset[i, 0]
            pos_i = _pos + offset[i, 1]

        vlc_i = bound_vlc(vlc_i)
        pos_i = bound_pos(pos_i)

        # convert state variables to feature indice
        f_i_y, f_i_x = featurize(vlc_i, pos_i)

        # add active feature's coordinate
        fs = np.append(fs, [int(f_i_y), int(f_i_x)])
    return fs

def get_action(_q, _w, _f_y, _f_x):

    q_t = _q[_f_y, _f_x]

    rand = np.random.rand()
    if rand >= epsilon:
        opt_a = np.where(q_t == np.max(q_t))[0]
        print(f'q_t={q_t}, opt_a={opt_a}')
        rand_i = np.random.randint(low=0, high=opt_a.size)
        a = opt_a[rand_i]
    else:
        a = np.random.randint(q_t.size)
    return a, _q

def get_next_state(_vlc, _pos, _a):
    next_vlc = _vlc + (0.001 * actions[_a]) - (0.0025 * math.cos(3 * _pos))
    next_pos = _pos + next_vlc

    next_vlc = bound_vlc(next_vlc)
    next_pos = bound_pos(next_pos)

    next_vlc = 0.0 if next_pos <= -1.2 else next_vlc
    return next_vlc, next_pos

def plot_step_cnt(_data, _x_limit, _labels):

    c = ['#555b6e','#c8553d', '#2a9d8f']
    x = np.arange(0, _x_limit, 1)

    for i in range(len(_labels)):
        plt.plot(x, _data[i], linewidth=0.5,label=f"α={_labels[i]}", c=c[i])

    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.title('Average steps per episode over 10 runs')
    plt.legend()
    plt.show()

def is_terminal(_pos):
    terminal = True if _pos >= 0.5 else False
    return terminal

def update_mean(_mean, _val, _cnt):
    if _cnt == 0:
        return 0
    return _mean + ((_val - _mean) / _cnt)

def animate_car(_ep, _pos_arr, _t, _title, _ax_title):

    fig = plt.figure()
    ax = plt.axes(xlim=(-1.2, 0.5), ylim=(0, 1.5))

    fig.suptitle(_title)
    ax.set_title(_ax_title)

    car, = ax.plot([], [], 'ro', ms=10, zorder=2.5)

    # plot mountain
    mt_inv = 100
    mt_x = np.linspace(-1.2, 0.5, mt_inv)
    mt_y = np.fromfunction(
        lambda self, n: (1.44087632 * (mt_x[n] + 0.35)) ** 2,
        (1, 100), dtype=int)[0]
    ax.plot(mt_x, mt_y, lw=0.8, zorder=0)

    def init():
        car.set_data([], [])
        return car,

    def animate(i):
        x = _pos_arr[i]
        y = (1.44087632 * (x + 0.35)) ** 2
        car.set_data(x, y)
        return car

    anim = animation.FuncAnimation(
        fig, animate, frames=_t, interval=5, blit=False, init_func=init)

    plt.show()

    # uncomment the line below to save the animation
    # anim.save(f'mt-car-sarsa-lambda_ep-{_ep}.mp4', writer='ffmpeg', fps=60)

def sarsa_lambda(_episodes, _alpha):

    # 8 @ 8 x 8 tilings
    # total features (x): 8 * 8 * 8 * 3 = 1536

    # Weight vector (w)
    w = np.zeros((8, 8, 8, 3))
    # Eligibility trace (x)
    z = np.zeros((8, 8, 8, 3))
    # action value
    q = np.zeros((8, 8, 3))

    for ep in range(_episodes):

        print(f'ep={ep}')
        pos_ep = np.array([])
        target_eps = np.array([0, 9, 49, 99])

        # initialize random pos and 0 velocity and time step
        pos = np.random.uniform(low=-0.6, high=-0.4)
        vlc, t = 0, 0
        f_y_p, f_x_p = featurize(vlc, pos)
        a_p, q = get_action(q, w, f_y_p, f_x_p)

        terminal = False
        while not terminal:

            pos_ep = np.append(pos_ep, pos)

            # choose epsilon-greedy action
            f_y, f_x = f_y_p, f_x_p
            a = a_p

            # get next state, check if terminal
            next_vlc, next_pos = get_next_state(vlc, pos, a)
            terminal = is_terminal(next_pos)

            # td_err = reward
            td_err = 0.0 if terminal else -1.0

            # get active features from all 8 tilings
            fs = generalize(vlc, pos)

            avg_w = 0.0
            for i in range(num_tilings):
                y_i, x_i = int(fs[i * 2]), int(fs[(i * 2) + 1])
                if x_i < 8 and y_i < 8:
                    avg_w += w[i, y_i, x_i, a]
                    z[i, y_i, x_i, a] += 1
            td_err -= (avg_w / num_tilings)

            if terminal:
                w += _alpha * td_err * z
                pos_ep = np.append(pos_ep, next_pos)
                continue

            q_t = est_q(w, fs, a)
            q[f_y, f_x, a] = q_t

            f_y_p, f_x_p = featurize(next_vlc, next_pos)
            next_fs = generalize(next_vlc, next_pos)

            a_p, q = get_action(q, w, f_y_p, f_x_p)

            avg_w = 0.0
            for i in range(num_tilings):
                y_p_i, x_p_i = int(next_fs[i * 2]), int(next_fs[(i * 2) + 1])
                if y_p_i < 8 and x_p_i < 8:
                    avg_w += w[i, y_p_i, x_p_i, a_p]

            td_err += gamma * (avg_w / num_tilings)
            w += _alpha * td_err * z
            z = gamma * lmbd * z

            vlc, pos = next_vlc, next_pos
            t += 1

        # episode terminated
        print(f't={t}')
        if ep in target_eps:
            title = f'Mountain Car with Sarsa(λ), λ={lmbd}, α={_alpha}, γ={gamma}'
            ax_title = f'Episode {ep + 1} finished in {t} steps'
            # animate_car(ep, pos_ep, t, title, ax_title)
            animate_car(ep, pos_ep, t, title, ax_title)

    # completed all episodes in for loop


if __name__ == '__main__':

    for i in range(alphas.size):

        alpha = alphas[i]

        for run in range(runs):
            print(f'run={run}')
            sarsa_lambda(episodes, alpha)