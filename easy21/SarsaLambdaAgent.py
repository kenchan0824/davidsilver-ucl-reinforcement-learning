import numpy as np

class SarsaLambdaAgent(object):

    def __init__(self, env, N0=100, gamma=1.0, lmbda=0.5):
        self.env = env
        self.N0 = N0
        self.gamma = gamma
        self.lmbda = lmbda
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

            E = np.zeros(self.env.STATE_DIM + (self.env.ACTION_DIM,))
            s = self.env.start()
            a = self.selectAction(s)

            while (not self.env.isTerminal(s)):

                r, s_ = self.env.step(s, a)
                self.N[tuple(s)+(a,)] += 1
                a_ = self.selectAction(s_)

                q = self.Q[tuple(s)+(a,)]
                q_ = self.Q[tuple(s_)+(a_,)]
                delta = (r + self.gamma * q_) - q
                E[tuple(s)+(a,)] += 1

                Alpha = np.zeros(self.env.STATE_DIM + (self.env.ACTION_DIM,))
                Alpha[self.N != 0] = 1.0 / self.N[self.N != 0]
                self.Q += Alpha * delta * E
                E *= self.gamma * self.lmbda

                (s, a) = (s_, a_)

        return self.Q



