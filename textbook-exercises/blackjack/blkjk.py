import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

class Casino:

    def __init__(self):
        self.player = Player()
        self.sum = np.array([0] * 10)
        self.usable_ace = {"exist": False, "pos": -1, "flipped_pos": -1}
        self.threshold = 17
        self.cards = np.append([1.0 / 13.0] * 9, (4.0 / 13.0))

    def clear_hand_n_sum(self):
        self.sum = np.array([0] * 10)
        self.usable_ace = \
            {"exist": False, "pos": -1, "flipped_pos": -1}

    def get_card(self):
        card = int(np.random.choice(10, 1, p=self.cards)[0]) + 1
        value = 11 if card == 1 else card
        return value

    def set_up_game(self):

        # clear trackers
        self.clear_hand_n_sum()
        self.player.clear_hand_n_sum()

        # deal two cards
        for i in range(2):
            self.hit(self, i)
            self.player.actions[i] = 1
            self.hit(self.player, i)


    def blkjk_td(self):

        # shorthand
        p = self.player
        self.set_up_game()

        i = 2
        end, d_stick, p_stick = \
            False, False, 0

        while not end:

            # check if dealer's threshold is met
            if self.sum[i - 1] < self.threshold and not d_stick:
                # hit
                self.hit(self, i)
            else:
                # stick
                self.sum[i] = self.sum[i - 1]
                d_stick = True

            # get player action
            if p.sum[i - 1] <= 21:
                hit = p.hit_or_stick(i, self.sum[0])
                if hit and not p_stick:
                    self.hit(p, i)
                    p.actions[i] = 1
                else:
                    p.sum[i] = p.sum[i - 1]
                    p.actions[i] = -1
                    p_stick += 1

            # dealer bust, player win
            if self.sum[i] > 21:

                p.sum[i] = 0
                p.actions[i] = 0
                p.td_update_v_q_pi(i, p_stick, 1.0, self.sum[0])
                end = True

            # player bust, player lose
            elif p.sum[i] > 21:

                p.td_update_v_q_pi(i, p_stick, -1.0, self.sum[0])
                end = True

            # check if both players will stick for next game
            elif d_stick and p_stick:

                # dealer wins
                if self.sum[i] > p.sum[i]:
                    p.td_update_v_q_pi(i, p_stick, -1.0, self.sum[0])

                # draw
                elif self.sum[i] == p.sum[i]:
                    p.td_update_v_q_pi(i, p_stick, 0.0, self.sum[0])

                # player wins
                else:
                    p.td_update_v_q_pi(i, p_stick, 1.0, self.sum[0])

                end = True

            # nothing happened. game goes on
            else:
                if p_stick == 0 or p_stick == 1:
                    p.td_update_v_q_pi(i, p_stick, 0.0, self.sum[0])

            i += 1


    def blkjk_mc(self):

        ## shorthand
        p = self.player
        self.set_up_game()

        i = 2
        end, d_stick, p_stick = \
            False, False, False

        while not end:

            # check if dealer's threshold is met
            if self.sum[i - 1] < self.threshold and not d_stick:
                # hit
                self.hit(self, i)
            else:
                # stick
                self.sum[i] = self.sum[i - 1]
                d_stick = True

            # get player action
            if p.sum[i - 1] <= 21:
                hit = p.hit_or_stick(i, self.sum[0])
                if hit and not p_stick:
                    self.hit(p, i)
                    p.actions[i] = 1
                else:
                    p.sum[i] = p.sum[i - 1]
                    p.actions[i] = -1
                    p_stick = True

            # check if someone bust
            if self.sum[i] > 21: # dealer bust, player win

                p.sum[i] = 0
                p.actions[i] = 0
                p.mc_update_v_q_pi(1.0, self.sum[0])
                end = True

            elif p.sum[i] > 21: # player bust, player lose

                p.mc_update_v_q_pi(-1.0, self.sum[0])
                end = True

            # check if both players will stick for next game
            elif d_stick and p_stick:

                # dealer wins
                if self.sum[i] > p.sum[i]:
                    p.mc_update_v_q_pi(-1.0, self.sum[0])
                # draw
                elif self.sum[i] == p.sum[i]:
                    p.mc_update_v_q_pi(0.0, self.sum[0])
                # player wins
                else:
                    p.mc_update_v_q_pi(1.0, self.sum[0])
                end = True

            i += 1


    # getting a new card, calculating and storing the new sum
    def hit(self, _p, _i):

        # getting a card
        card = self.get_card()

        # storing data if card is an usable ace
        if card == 11:
            _p.usable_ace['exist'] = True
            _p.usable_ace['pos'] = _i

        # checking bust and adjusting ace value
        sum_i = card if _i == 0 \
            else (_p.sum[_i - 1] + card)

        if sum_i > 21 and _p.usable_ace['exist']:
            if (sum_i - 10) <= 21:
                sum_i -= 10
                _p.usable_ace['exist'] = False
                _p.usable_ace['flipped_pos'] = _i

        _p.sum[_i] = sum_i


class Player:

    def __init__(self, _eps=0.01, _gamma=1.0, _alpha=0.1):

        self.eps = _eps
        self.gamma = _gamma
        self.alpha = _alpha

        self.sum = np.array([0] * 10)
        self.actions = np.array([0] * 10)
        self.usable_ace = {'exist': False, 'pos': -1, 'flipped_pos': -1}

        # [cnt, v(s), cnt(a1), q(s, a1), q(s, a2)]
        self.no_ace = np.full((10, 10, 5), 0.0)
        self.ace = np.full((10, 10, 5), 0.0)

        # [no ace policy, ace policy]
        self.pi = np.full((10, 10, 2), True)

        for i in range(10):
            for j in range(10):
                for k in range(2):
                    if j > 7:
                        self.pi[i, j, k] = False



    def clear_hand_n_sum(self):
        self.sum = np.array([0] * 10)
        self.actions = np.array([0] * 10)
        self.usable_ace = \
            {"exist": False, "pos": -1, "flipped_pos": -1}

    def hit_or_stick(self, _i, _fc):

        sum = self.sum[_i - 1]

        if sum == 21 or sum == 20: # hard stick
            return False
        elif sum < 12:  # hard hit
            return True
        else:
            rand = np.random.rand(1)[0]

            pos = 1 if self.usable_ace['exist'] else 0
            fc = 0 if _fc == 11 else _fc - 1

            if self.pi[fc, sum - 12, pos]: # hit
                if rand >= self.eps:  # exploit: hit
                    return True
                else:  # explore: stick
                    return False

            else: # stick with epsilon
                if rand >= self.eps:  # exploit: stick
                    return False
                else:  # explore: hit
                    return True


    def plot_pi(self, _episode):

        def plot(_ax, _k):
            for i in range(10):
                for j in range(10):
                    mkr = 'go' if self.pi[x[i] - 1, y[j] - 12, _k] else 'rX'
                    _ax.plot(x[i], y[j], mkr)

        # dealer card shown
        x = np.zeros(10, dtype=int)
        for i in range(1, 11):
            x[i - 1] = i

        # player's sum
        y = np.arange(12, 22)

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3,
            sharex='all', sharey='all',
            gridspec_kw={'width_ratios': [5, 5, 1]}
        )
        ax3.axis('off')

        plot(ax1, 1)
        plot(ax2, 0)

        fig.suptitle(f'Policy after {_episode} episodes, ε={self.eps}, γ={self.gamma}')
        ax1.set_title('Usable ace')
        ax1.set_xlabel('Dealer card shown')
        ax1.set_ylabel('Player sum')
        ax2.set_title('No usable ace')
        ax2.set_xlabel('Dealer card shown')
        ax2.set_ylabel('Player sum')

        plt.xticks(x)
        plt.yticks(y)

        legend_el = [
            Line2D([0], [0], marker='o', color='w', label='Hit',
                          markerfacecolor='g', markersize=15),
            Line2D([0], [0], marker='X', color='w', label='Stick',
                   markerfacecolor='r', markersize=15),
        ]
        ax3.legend(handles=legend_el, loc='center')
        plt.show()

    def plot_v(self, _episode):

        def f(_arr, _x, _y):
            return np.asarray(
                [[_arr[x - 1, y - 12, 1] for x in _x] for y in _y]
            )

        # dealer card shown
        x = np.zeros(10, dtype=int)
        for i in range(1, 11):
            x[i - 1] = i

        # player's sum
        y = np.arange(12, 22)

        z1 = f(self.ace, x, y)
        z2 = f(self.no_ace, x, y)


        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3,
            sharex='all', sharey='all',
            gridspec_kw={'width_ratios': [5, 5, 1]},
            subplot_kw={"projection": "3d"}
        )
        ax3.axis('off')

        ax1.plot_surface(
            X=[[X for X in x] for i in range(y.size)],
            Y=[[Y for j in range(x.size)] for Y in y],
            Z=z1,
            cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax2_plot = \
            ax2.plot_surface(
                X=[[X for X in x] for i in range(y.size)],
                Y=[[Y for j in range(x.size)] for Y in y],
                Z=z2,
                cmap=cm.coolwarm, linewidth=0, antialiased=False)

        fig.suptitle(f'episode={_episode}, ε={self.eps}, γ={self.gamma}')
        ax1.set_title('Usable ace')
        ax1.set_xlabel('Dealer card shown')
        ax1.set_ylabel('Player sum')
        ax2.set_title('No usable ace')
        ax2.set_xlabel('Dealer card shown')
        ax2.set_ylabel('Player sum')
        ax1.set_zlim(-1, 1)
        ax2.set_zlim(-1, 1)

        plt.subplots_adjust(left=1/30)
        cbaxes = fig.add_axes([0.86, 0.15, 0.03, 0.7])
        plt.colorbar(ax2_plot, cax=cbaxes)
        plt.show()


    def update_mean(self, _mean, _val, _cnt):
        if _cnt == 0:
            return 0
        return _mean + ((_val - _mean) / _cnt)


    def td_update_v_q_pi(self, _i, _stick, _r, _fc):

        # adjusting dealer's first card value for array indexing
        _fc = 0 if _fc == 11 else _fc - 1

        v = self.ace if self.usable_ace['exist'] else self.no_ace
        ace = 1 if self.usable_ace['exist'] else 0
        s = self.sum[_i - 1] - 12
        sp = self.sum[_i] - 12

        if (s + 12 >= 12 or s + 12 <= 21) and (sp + 12 >= 12):

            # update v
            print(f's = {s}')
            print(f'sp = {sp}')
            v_sp = -1.0 if sp + 12 > 21 else v[_fc, sp, 1]

            v[_fc, s, 0] += 1.0
            v[_fc, s, 1] = self.update_mean(
                _mean=v[_fc, s, 1],
                _val=self.alpha * (_r + (self.gamma * v_sp) - v[_fc, s, 1]),
                _cnt=v[_fc, s, 0]
            )

            # update q
            if sp + 12 > 21:
                sum_sp_ap = 0
            else:
                sp_hit_eps = (1 - self.eps) if self.pi[_fc, sp, ace] else self.eps
                sp_stk_eps = self.eps if self.pi[_fc, sp, ace] else (1 - self.eps)

                sum_sp_ap = (sp_hit_eps * v[_fc, sp, 3]) + (sp_stk_eps * v[_fc, sp, 4])

            if _stick:

                q = self.alpha * (_r + (self.gamma * sum_sp_ap - v[_fc, s, 4]))

                v[_fc, s, 4] = self.update_mean(
                    _mean=v[_fc, s, 4],
                    _val=q,
                    _cnt=v[_fc, s, 0] - v[_fc, s, 2]
                )

            else:

                q = self.alpha * (_r + (self.gamma * sum_sp_ap - v[_fc, s, 3]))

                v[_fc, s, 2] += 1.0
                v[_fc, s, 3] = self.update_mean(
                    _mean=v[_fc, s, 3],
                    _val=q,
                    _cnt=v[_fc, s, 2]
                )

            # update policy
            if s + 12 <= 19:
                self.pi[_fc, s, ace] = True if v[_fc, s, 3] >= v[_fc, s, 4] else False




    def mc_update_v_q_pi(self, _r, _fc):

        # adjusting dealer's first card value for array indexing
        _fc = 0 if _fc == 11 else _fc - 1

        # init G
        G = 0.0

        # clean up array for episode
        episode = self.sum[self.sum != 0]
        actions = self.actions[self.actions != 0]

        stick = False

        for i in range(episode.size - 1):

            t = episode.size - 1 - i

            if self.usable_ace['exist']:
                ace = 1
                v = self.ace
            else:
                ace = 1 \
                    if self.usable_ace['pos'] <= t < self.usable_ace['flipped_pos'] \
                    else 0
                v = self.ace if ace else self.no_ace

            s = self.sum[t] - 12

            r = _r if i == 0 else 0.0
            G = self.gamma * G + r

            if s + 12 < 12 or s + 12 > 21:
                continue

            # update v
            v[_fc, s, 0] += 1.0
            v[_fc, s, 1] = self.update_mean(
                _mean=v[_fc, s, 1],
                _val=G,
                _cnt=v[_fc, s, 0]
            )

            if actions[t] < 0 and not stick: # update q for stick
                stick = True
                v[_fc, s, 4] = self.update_mean(
                    _mean=v[_fc, s, 4],
                    _val=G,
                    _cnt=v[_fc, s, 0] - v[_fc, s, 2]
                )
            elif actions[t] > 0: # update q for hit
                v[_fc, s, 2] += 1.0
                v[_fc, s, 3] = self.update_mean(
                    _mean=v[_fc, s, 3],
                    _val=G,
                    _cnt=v[_fc, s, 2]
                )

            if s + 12 > 19:
                continue

            # update policy
            self.pi[_fc, s, ace] = True if v[_fc, s, 3] > v[_fc, s, 4] else False


if __name__ == '__main__':

    episodes = 1000000
    casino = Casino()

    for i in range(episodes):
        casino.blkjk_td()

    casino.player.plot_v(episodes)
    casino.player.plot_pi(episodes)
