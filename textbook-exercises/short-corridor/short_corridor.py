import math
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    runs = 100

    alpha = 0.003
    gamma = 0.99

    # [right , left]
    actions = np.array([[1, 0], [-1, 1], [1, -1]])
    x = np.array([[1, 0], [0, 1]])
    w = np.array([0.0, 0.0])

    avg_pi = np.array([0.0, 0.0])
    avg_return = np.array([0.0, 0.0])
    avg_partial_t = 0


    for run in range(runs):

        print(f'_RUN_{run + 1}_')
        theta = np.array([0.0, 0.0])

        pi = np.array([0.0, 0.0])
        q = np.array([0.0, 0.0])
        a_record = np.array([])

        # pi for this episode
        h0 = math.exp(np.sum(theta * x[0]))
        h1 = math.exp(np.sum(theta * x[1]))
        pi_ep = np.array([h0 / (h0 + h1), h1 / (h0 + h1)])

        s = 0
        t = 0

        terminal = False
        while not terminal:

            t += 1

            # select an action
            a = np.random.choice([0, 1], p=pi_ep)
            a_record = np.append(a_record, a)

            sp = s + actions[s, a]

            if sp >= 3:
                terminal = True
                continue

            s = sp

        print(f"t={t}, a_record={a_record}")
        G = 0

        for i in range(t):
            G_t = 0
            pw = 0
            for j in range(i, t):
                r = 0 if (j == i) else -1
                G_t += (gamma ** pw) * -1
                pw += 1

            if i == 0:
                G = G_t

            ai = int(a_record[i])
            for j in range(theta.size):
                theta[j] += alpha * (gamma ** i) * G_t * (1 / pi_ep[ai]) * x[ai, j]
            print(f'theta={theta}')

            h_ai = theta[ai]
            pi[ai] = 1 / (1 + np.exp(h_ai))
            pi[1 - ai] = 1 - pi[ai]
            print(f'pi={pi}')

        for k in range(avg_pi.size):
            avg_pi[k] += (pi[k] - avg_pi[k]) / (run + 1)
            avg_return[k] += (G - avg_return[k]) / (run + 1)

    print('\nSHORT CORRIDOR RESULTS -----------------------------------------------------------------')
    print(f'{runs} runs, α={alpha}, γ={gamma}')
    print(f'avg_pi={avg_pi}, avg_return={avg_return}')