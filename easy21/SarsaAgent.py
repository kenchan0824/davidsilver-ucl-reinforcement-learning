import numpy as np

class SarsaAgent(object):

    def __init__(self, env, N0=100, gamma=1.0):
        self.env = env
        self.N0 = N0
        self.gamma = gamma
        self.N = np.zeros(env.STATE_DIM + (env.ACTION_DIM,))
        self.Q = np.zeros(env.STATE_DIM + (env.ACTION_DIM,))

    def selectAction(self, s):
        epsilon = self.N0 / (self.N0 + np.sum(self.N[s.index()]))
        if np.random.rand() < (1 - epsilon):  # choose greedy action
            a = np.argmax(self.Q[s.index()])
        else:
            a = np.random.randint(self.env.ACTION_DIM)

        return a

    def value(self):
        V = np.max(self.Q, axis=2)
        return V

    def learn(self, nEpisodes):

        for k in range(nEpisodes):

            s = self.env.start()
            a = self.selectAction(s)
            while (not s.isTerminal()):

                r, s_ = self.env.step(s, a)
                self.N[s.index()+(a,)] += 1
                a_ = self.selectAction(s_)

                q = self.Q[s.index()+(a,)]
                q_ = self.Q[s_.index()+(a_,)]
                delta = (r + self.gamma * q_) - q
                alpha = 1.0 / self.N[s.index()+(a,)]
                self.Q[s.index()+(a,)] += alpha * delta

                (s, a) = (s_, a_)

        return self.Q



