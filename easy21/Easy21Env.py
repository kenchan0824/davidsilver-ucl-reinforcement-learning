import numpy as np

class Easy21State(object):

    def __init__(self, player=None, dealer=None):
        deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cards = np.random.choice(deck, 2, replace=True)
        self.player = [cards[0]] if player is None else player
        self.dealer = [cards[1]] if dealer is None else dealer

    def copy(self):
        s = Easy21State(self.player.copy(), self.dealer.copy())
        return s

    def drawCard(self):
        deck = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*2 + [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
        card = np.random.choice(deck, replace=True)
        return card

    def isBurst(self, hand):
        sumHand = sum(hand)
        if (sumHand  > 21 or sumHand < 1):
            return True

        return False

    def isTerminal(self):
        if self.isBurst(self.player) or self.isBurst(self.dealer):
            return True
        playerSum = sum(self.player)
        dealerSum = sum(self.dealer)
        if dealerSum >= 17 and dealerSum > playerSum:   # dealer hit until win or burst
            return True

        return False

    def reward(self):
        if not self.isTerminal():
            return 0
        elif self.isBurst(self.player):
            return -1
        elif self.isBurst(self.dealer):
            return 1
        elif sum(self.dealer) > sum(self.player):
            return -1

        return 0

    def index(self):
        if self.isTerminal():
            return 0, 0         # return a single index for all terminal states

        return (sum(self.player), sum(self.dealer))

    def hit(self):
        card = self.drawCard()
        self.player.append(card)

    def stick(self):
        while not self.isTerminal():
            card = self.drawCard()
            self.dealer.append(card)


class Easy21Env(object):

    STATE_DIM = (22, 11)
    ACTION_DIM = 2

    def start(self):
        s = Easy21State()
        return s

    def step(self, s, a):
        if (s.isTerminal()):
            return 0, s

        s_ = s.copy()
        # player turn
        if (a == 1):  # hit
            s_.hit()
        # dealer turn
        elif (a == 0):  # stick
            s_.stick()

        r = s_.reward()
        return r, s_
