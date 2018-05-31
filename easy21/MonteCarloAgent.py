import numpy as np

class MonteCarloAgent(object):

    def __init__(self, env, N0):
        self.env = env
        self.N0 = N0
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

            visits = []
            G = 0
            s = self.env.start()

            while (not self.env.isTerminal(s)):

                a = self.selectAction(s)
                self.N[tuple(s)+(a,)] += 1
                visits.append([s, a])

                r, s = self.env.step(s, a)
                G += r

            for [s, a] in visits:

                alpha = 1/ self.N[tuple(s)+(a,)]
                self.Q[tuple(s)+(a,)] += alpha * (G - self.Q[tuple(s)+(a,)])

        return self.Q



