import math
import numpy as np
import matplotlib.pyplot as plt

# environment
states = \
    np.array(['T1', 'A', 'B', 'C', 'D', 'E', 'T2'])
true_v = \
    np.array([0.0, (1.0/6.0), (2.0/6.0), (3.0/6.0), (4.0/6.0), (5.0/6.0), 0.0])

def get_reward(_s):
    return 1 if states[_s] == 'T2' else 0

def is_terminal(_s):
    return True if _s == 0 or _s == states.size - 1 else False

def get_rms(_v):
    err_sum = 0
    for i in range(1, _v.size - 1):
        err_sum += (_v[i] - true_v[i]) ** 2
    return math.sqrt(err_sum / (_v.size - 2))

def plot_error(_episodes, _err):
    x = np.arange(_episodes)

    for i in range(alphas.size):
        y = _err[i]
        plt.plot(x, y, linewidth=1.0)

    plt.legend(alphas, title='Alpha value', loc="upper right")
    plt.title('Random Walk using TD(0), v(s) initialized as 4/6')
    plt.xlabel("Walks(episodes)")
    plt.ylabel("Empirical RMSE averaged over states")
    plt.show()

def walk(_s):
    if np.random.rand() >= 0.5:
        return _s - 1
    return _s + 1

def update_err_per_step(_err, _v, _step):
    for i in range(_err.size):
        _err[i] = _err[i] + ((_v[i + 1] - true_v[i + 1])/_step)
    return _err

def update_mean(_mean, _val, _cnt):
    return _mean + ((_val - _mean) / _cnt)

def update_v(_alpha, _gamma, _s, _sp, _r):
    v[_s] = v[_s] + _alpha * (_r + (_gamma * v[_sp]) - v[_s])

# agent
run = 100
episode = 100
gamma = 1.0
alphas = np.array([0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.01])

err_over_runs = np.zeros((alphas.size, episode))
for n in range(alphas.size):

    alpha = alphas[n]

    for i in range(run):

        v = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])

        for j in range(episode):

            s_err = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            current_state = 3

            while not is_terminal(current_state):
                # get a random step
                next_state = walk(current_state)

                # get reward of s'
                reward = get_reward(next_state)

                # update state-value of s
                # based on reward and state-value of s'
                update_v(alpha, gamma, current_state, next_state, reward)

                # take the step
                current_state = next_state

            err_over_runs[n, j] = \
                update_mean(err_over_runs[n, j], get_rms(v), j + 1)

print(err_over_runs)
plot_error(episode, err_over_runs)


