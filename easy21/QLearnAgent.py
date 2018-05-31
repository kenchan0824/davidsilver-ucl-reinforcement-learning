import numpy as np

class QLearnAgent(object):

    def __init__(self, env, N0=100, gamma=1.0):
        self.env = env
        self.N0 = N0
        self.gamma = gamma
        self.N = np.zeros(env.STATE_DIM + (env.ACTION_DIM,))
        self.Q = np.zeros(env.STATE_DIM + (env.ACTION_DIM,))

    def selectAction(self, s):
        epsilon = self.N0 / (self.N0 + np.sum(self.N[tuple(s)]))
        if np.random.rand() < (1 - epsilon):  # choose greedy action
            a = np.argmax(self.Q[tuple(s)])
        else:
            a = np.random.randint(self.env.ACTION_DIM)
        return a

    def value(self):
        V = np.max(self.Q, axis=2)
        return V

    def learn(self, nEpisodes):

        for k in range(nEpisodes):

            s = self.env.start()
            while (not self.env.isTerminal(s)):
                a = self.selectAction(s)
                r, s_ = self.env.step(s, a)
                self.N[tuple(s)+(a,)] += 1

                q = self.Q[tuple(s)+(a,)]
                q_max = np.max(self.Q[tuple(s_)])
                delta = r + self.gamma * q_max - q
                alpha = 1.0 / self.N[tuple(s)+(a,)]
                self.Q[tuple(s)+(a,)] += alpha * delta

                s = s_

        return self.Q



