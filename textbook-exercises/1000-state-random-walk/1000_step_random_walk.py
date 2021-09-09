import math
import numpy as np
import matplotlib.pyplot as plt

STATE_NUM = 1000
GROUP_NUM = 10
RUNS = 100
START = 500
END_L = 0
END_R = 1001

ALPHA = 0.8

class RandWalk_1000L:

    def __init__(self, _alpha=ALPHA, _runs=RUNS, _state_num=STATE_NUM):
        self.alpha = _alpha
        self.runs = _runs
        self.state_num = _state_num
        self.state = START

        self.weights = np.zeros(GROUP_NUM)


    def get_sp_r(self):

        # get a random step size from -100 to 100
        # (both inclusive) but excluding 0
        steps = 0
        while steps == 0:
            steps = np.random.randint(low=-100, high=101)

        sp = self.state + steps
        sp = END_R if sp >= END_R else sp
        sp = END_L if sp <= END_L else sp

        r = 0
        r = 1 if sp == END_R else r
        r = -1 if sp == END_L else r

        return sp, r

    def walk_TD_n(self, _n):

        for run in range(self.runs):

            weights_run = np.zeros(GROUP_NUM)

            for ep in range(100):

                self.state = START
                s_ep = np.array([START])
                r_ep = np.array([0])
                T = 10000
                t = 0
                ro = 0
                while ro != T - 1:

                    while t < T:
                        sp, r = self.get_sp_r()
                        np.append(s_ep, sp)
                        np.append(r_ep, r)
                        T = t + 1 if sp == END_R or sp == END_L else T
                        self.state = sp

                    ro = t - _n + 1




    def walk_TD_0(self):

        for run in range(self.runs):

            weights_run = np.zeros(GROUP_NUM)

            for ep in range(100):

                self.state = START
                terminal = False
                while not terminal:

                    sp, r = self.get_sp_r()

                    # set terminal boolean
                    terminal = True if r != 0 else terminal

                    # STATE AGGREGATION                                ( P.203 in textbook )#
                    # is a simple form of generalizing function approximation in which      #
                    # states are grouped together, with one estimated value (one component  #
                    # of the weight vector w for each group. The value of a state is        #
                    # estimated as its group’s component, and when the state is updated,    #
                    # that component alone is updated. State aggregation is a special case `#
                    # of SGD in which the gradient is 1 for S_t's group's component and`    #
                    # 0 for other components.                                               #

                    gradient = 1
                    index_s = (self.state - 1) // 100
                    index_sp = (sp - 1) // 100

                    v_sp = 0 if terminal else weights_run[index_sp]

                    weights_run[index_s] += \
                        self.alpha * (r + v_sp - weights_run[index_s]) * gradient

                    # move to next state
                    self.state = sp

            # store weights this run into weights over runs
            for i in range(self.weights.size):
                self.weights[i] += (weights_run[i] - self.weights[i]) / (run + 1)

        print(self.weights)


if __name__ == '__main__':

    rw = RandWalk_1000L()
    rw.walk_TD_0()




