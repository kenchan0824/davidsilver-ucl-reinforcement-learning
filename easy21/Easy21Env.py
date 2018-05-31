import numpy as np

class Easy21Env(object):

    STATE_DIM = (22, 22)
    ACTION_DIM = 2

    def isBurst(self, s, pos):
        if (s[pos] == 0 or s[pos] > 21 or s[pos] < 1):
            s[pos] = 0
            return True
        return False

    def isTerminal(self, s):
        if (self.isBurst(s,0) or self.isBurst(s,1)):
            return True
        elif (s[1] >= 17):
            return True
        return False

    def reward(self, s):
        if not self.isTerminal(s):
            return 0
        elif self.isBurst(s, 0):
            return -1
        elif self.isBurst(s, 1):
            return 1
        elif s[0] < s[1]:
            return -1
        elif s[0] > s[1]:
            return 1
        return 0

    def start(self):
        cards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        s = np.random.choice(cards, 2, replace=True)
        return s

    def step(self, s, a):

        if (self.isTerminal(s)):
            return 0, s

        s_ = s.copy()
        cards = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*2 + [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])

        # player turn
        if (a == 1):  # hit
            newcard = np.random.choice(cards, replace=True)
            s_[0] += newcard
            self.isBurst(s, 0)

        # dealer turn
        elif (a == 0):  # stick
            while (not self.isTerminal(s_)):
                newcard = np.random.choice(cards, replace=True)
                s_[1] += newcard

        r = self.reward(s_)
        return r, s_
