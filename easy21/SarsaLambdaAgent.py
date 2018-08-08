import numpy as np

class SarsaLambdaAgent(object):

    def __init__(self, env, N0=100, gamma=1.0, lmbda=0.9):
        self.env = env
        self.N0 = N0
        self.gamma = gamma
        self.lmbda = lmbda
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
        V = np.max(self.Q, axis=-1) # -1 for last axis (action)
        return V

    def learn(self, nEpisodes):

        for k in range(nEpisodes):

            E = np.zeros(self.env.STATE_DIM + (self.env.ACTION_DIM,))
            s = self.env.start()
            a = self.selectAction(s)

            while (not s.isTerminal()):

                r, s_ = self.env.step(s, a)
                self.N[s.index()+(a,)] += 1
                a_ = self.selectAction(s_)

                q = self.Q[s.index()+(a,)]
                q_ = self.Q[s_.index()+(a_,)]
                delta = (r + self.gamma * q_) - q
                E[s.index()+(a,)] += 1

                Alpha = np.zeros(self.env.STATE_DIM + (self.env.ACTION_DIM,))
                Alpha[self.N != 0] = 1.0 / self.N[self.N != 0]
                self.Q += Alpha * delta * E
                E *= self.gamma * self.lmbda

                (s, a) = (s_, a_)

        return self.Q

    def simulate(self, nSim):

        G = np.zeros(nSim)
        for k in range(nSim):

            for i in range(100):

                s = self.env.start()
                while (not s.isTerminal()):
                    a = np.argmax(self.Q[s.index()])
                    r, s_ = self.env.step(s, a)
                    G[k] += r

                    s = s_

        return G.mean(), 1.96 * G.std() / np.sqrt(nSim)
