import math
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    max_t = 1000

    # feature vector
    x = np.array([
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    ])
    # weight vector for 8 components
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
    record_w = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [10.0], [1.0]])

    # alpha / |S|
    alpha = 0.014 / 7.0

    # discounted variable
    gamma = 0.99

    # init state at random
    s = np.random.randint(7)

    # start of episode
    for t in range(max_t):

        sp = np.random.randint(7)

        if sp == 6:

            # get current state's value
            v = (x[s]) @ w

            # get future state's value
            vp = (x[sp]) @ w

            w[s] = w[s] + alpha * (7.0 * ((gamma * vp) - v)) * 3.0
            w[7] = w[7] + alpha * (7.0 * ((gamma * vp) - v)) * 3.0

        s = sp
        record_w = np.insert(record_w, record_w.shape[1], w, axis=1)

    print(record_w)
    x = np.arange(0, max_t + 1)
    for i in range(w.size):
        plt.plot(x, record_w[i], linewidth=1.0, label=f'w{i+1}')
    plt.xlabel('steps')
    plt.title("Baird's Counterexample with Semi-gradient Off-policy TD")
    plt.legend()
    plt.show()

