import math
import numpy as np
import matplotlib.pyplot as plt

runs = 10
episodes = 10

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
alphas = \
    np.array([(0.5 / num_tilings_f)])
epsilon = 0.0
gamma = 0.98

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
    q = 0
    for i in range(num_tilings):
        y_i, x_i = int(_fs[i * 2]), int(_fs[(i * 2) + 1])
        if x_i < 8 and y_i < 8:
            q += _w[i, y_i, x_i, _a]
    return q

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
                vlc_i = _vlc + (np.random.randint(low=-1, high=2) * y_unit)
                pos_i = _pos + (np.random.randint(low=-1, high=2) * x_unit)
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

def get_action(_q, _f_y, _f_x):
    q_t = _q[_f_y, _f_x]
    rand = np.random.rand()
    if rand >= epsilon:
        opt_a = np.where(q_t == np.max(q_t))[0]
        a = opt_a[np.random.randint(opt_a.size)]
    else:
        a = np.random.randint(q_t.size)
    return a

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

def sarsa_0(_run_i, _episodes, _avg_t_i, _alpha):

    avg_t_run = 0

    # 3 action-values per state: [0, forward, reverse]
    q = np.zeros((8, 8, 3))

    # 8 @ 8 x 8 tilings
    # total features (x): 8 * 8 * 8 * 3 = 1536
    w = np.zeros((8, 8, 8, 3))

    for ep in range(_episodes):

        print(f'ep={ep}')

        # initialize random pos and 0 velocity and time step
        pos = np.random.uniform(low=-0.6, high=-0.4)
        vlc, t = 0, 0
        f_y_p, f_x_p = featurize(vlc, pos)
        a_p = get_action(q, f_y_p, f_x_p)

        terminal = False
        while not terminal:

            # choose epsilon-greedy action
            f_y, f_x = f_y_p, f_x_p
            a = a_p

            # get next state, check if terminal
            next_vlc, next_pos = get_next_state(vlc, pos, a)
            terminal = is_terminal(next_pos)

            if terminal:
                q_p = 0
                r = 0
            else:
                # get next action
                f_y_p, f_x_p = featurize(next_vlc, next_pos)
                a_p = get_action(q, f_y_p, f_x_p)

                # get next generalized feature vector
                next_features = generalize(next_vlc, next_pos)
                q_p = est_q(w, next_features, a_p)
                q[f_y_p, f_x_p, a_p] = q_p

                r = -1

            features = generalize(vlc, pos)
            q_t = est_q(w, features, a)
            q[f_y, f_x, a] = q_t

            # update weights vector
            for j in range(8):
                w[j, f_y, f_x, a] = w[j, f_y, f_x, a] + \
                                    (_alpha * (r + (gamma * q_p) - q_t)) * 1
            vlc, pos = next_vlc, next_pos
            t += 1

        # episode terminated
        avg_t_run = update_mean(avg_t_run, t, ep + 1)
        _avg_t_i[ep] = update_mean(_avg_t_i[ep], t, _run_i + 1)

    # end of run
    return q, w, _avg_t_i

def sarsa_n(_n, _run_i, _episodes, _avg_t_i, _alpha):

    avg_t_run = 0

    # 3 action-values per state: [0, forward, reverse]
    q = np.zeros((8, 8, 3))

    # 8 @ 8 x 8 tilings
    # total features (x): 8 * 8 * 8 * 3 = 1536
    w = np.zeros((8, 8, 8, 3))

    for ep in range(_episodes):

        print(f'ep={ep}')

        # INITIALIZE
        # Random pos and 0 velocity and time step
        pos = np.random.uniform(low=-0.6, high=-0.4)
        vlc, t = 0, 0

        # First state-action pair
        f_y, f_x = featurize(vlc, pos)
        a = get_action(q, f_y, f_x)

        # INITIALIZE
        # memory for state-action sequence
        # as an array of active features:
        # [vlc, pos]with size of (8, 2)
        fs_seq = np.array([generalize(vlc, pos)])
        a_seq = np.array([a])
        r_seq = np.array([])

        T = 100000000
        while True:

            if t < T:

                # get next state, check if terminal
                next_vlc, next_pos = get_next_state(vlc, pos, a)

                if is_terminal(next_pos):
                    T = t + 1
                    r_seq = np.append(r_seq, 0)

                else:
                    f_y_p, f_x_p = featurize(next_vlc, next_pos)
                    fs_p = generalize(next_vlc, next_pos)
                    a_p = get_action(q, f_y_p, f_x_p)

                    fs_seq = np.append(fs_seq, fs_p)
                    r_seq = np.append(r_seq, -1)
                    a_seq = np.append(a_seq, a_p)

                    vlc, pos = next_vlc, next_pos
                    a = a_p

            tau = t - _n + 1
            if tau >= 0:

                end = min(tau + _n, T)
                G = 0
                for i in range(tau + 1, end):
                    G += (gamma ** (i - tau - 1)) * r_seq[i]

                if tau + _n < T:
                    fs_tn = fs_seq[16 * (tau + _n):(16 * (tau + _n)) + 16]
                    a_tn = a_seq[tau + _n]
                    G += (gamma ** _n) * est_q(w, fs_tn, a_tn)

                fs_tau = fs_seq[16 * tau: (16 * tau) + 16]
                a_tau = a_seq[tau]
                q_tau = est_q(w, fs_tau, a_tau)

                # update weights vector
                for j in range(8):
                    y_j, x_j = int(fs_tau[j * 2]), int(fs_tau[(j * 2) + 1])
                    w[j, y_j, x_j, a_tau] += (_alpha * (G - q_tau)) * 1

            t += 1

            if tau == T - 1:
                break

        # episode terminated
        avg_t_run = update_mean(avg_t_run, t, ep + 1)
        _avg_t_i[ep] = update_mean(_avg_t_i[ep], t, _run_i + 1)

    # end of run
    return q, w, _avg_t_i


if __name__ == '__main__':

    avg_t = np.zeros((alphas.size, episodes))
    avg_q = np.zeros((alphas.size, 8, 8, 3))

    for i in range(alphas.size):

        alpha = alphas[i]
        avg_t_i = np.zeros(episodes)
        avg_q_i = np.zeros((8, 8, 3))

        for run in range(runs):
            print(f'run={run}')
            q, w, avg_t_i = sarsa_n(4, run, episodes, avg_t_i, alpha)

        # end of run
        for ep in range(episodes):
            avg_t[i, ep] = avg_t_i[ep]

    plot_step_cnt(avg_t, episodes, alphas)