import numpy as np

class QLearnAgent:

    def __init__(self, env, N0=100, gamma=1.0):
        
        self.env = env
        self.N0 = N0
        self.gamma = gamma
        self.N = np.zeros(env.STATE_DIM + (env.ACTION_DIM,))
        self.Q = np.zeros(env.STATE_DIM + (env.ACTION_DIM,))
        

    def selectAction(self, s):
        
        epsilon = self.N0 / (self.N0 + np.sum(self.N[s]))
        if np.random.rand() < (1 - epsilon):  # choose greedy action
            a = np.argmax(self.Q[s])
        else:
            a = np.random.randint(self.env.ACTION_DIM)
            
        return a


    def value(self):

        V = np.max(self.Q, axis=-1) # -1 for last axis (action)
        return V


    def learn(self, nEpisodes):

        for k in range(nEpisodes):

            s = self.env.reset()
            done = False
            while not done:
                a = self.selectAction(s)
                s_, r, done = self.env.step(a)
                self.N[s+(a,)] += 1
                q = self.Q[s+(a,)]
                
                if not done:
                    q_max = np.max(self.Q[s_])
                    delta = r + self.gamma * q_max - q
                else:
                    delta = r - q
                    
                alpha = 1.0 / self.N[s+(a,)]
                self.Q[s+(a,)] += alpha * delta

                s = s_

        return self.Q


    def simulate(self, nSim):

        G = np.zeros(nSim)
        for k in range(nSim):

            for i in range(100):

                s = self.env.reset()
                done = False
                while not done:
                    
                    a = np.argmax(self.Q[s])
                    s_, r, done = self.env.step(a)
                    G[k] += r

                    s = s_

        return G.mean(), 1.96 * G.std() / np.sqrt(nSim)
