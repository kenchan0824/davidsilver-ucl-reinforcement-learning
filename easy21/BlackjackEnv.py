import numpy as np

class BlackjackState(object):

    def __init__(self, nDeck=None, player=None, dealer=None):
        self.nDeck = nDeck
        self.player = [self.drawCard()] if player is None else player
        self.dealer = [self.drawCard()] if dealer is None else dealer

    def copy(self):
        s = BlackjackState(self.nDeck, self.player.copy(), self.dealer.copy())
        return s

    def drawCard(self):
        deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
                        *(1 if self.nDeck is None else self.nDeck))
        card = np.random.choice(deck, replace=(self.nDeck is None))
        return card

    def usableAce(self, hand):
        if (1 in hand) and (sum(hand) < 12):
            return True

        return False

    def sumHand(self, hand):
        sumHand = sum(hand)
        if self.usableAce(hand):
            sumHand += 10

        return sumHand

    def isTerminal(self):
        playerSum = self.sumHand(self.player)
        dealerSum = self.sumHand(self.dealer)
        if playerSum > 21 or dealerSum > 21:
            return True
        elif dealerSum >= 17 and dealerSum > playerSum:
            return True

        return False

    def reward(self):
        if not self.isTerminal():
            return 0

        playerSum = self.sumHand(self.player)
        dealerSum = self.sumHand(self.dealer)

        if playerSum > 21:
            return -1
        elif dealerSum > 21:
            return 1
        elif dealerSum > playerSum:
            return -1

        return 0

    def index(self):
        if self.isTerminal():
            return 0, 0, 0

        return (self.sumHand(self.player),
                self.sumHand(self.dealer),
                int(self.usableAce(self.player)))

    def hit(self):
        card = self.drawCard()
        self.player.append(card)

    def stick(self):
        while not self.isTerminal():
            card = self.drawCard()
            self.dealer.append(card)


class BlackjackEnv(object):

    STATE_DIM = (22, 12, 2)
    ACTION_DIM = 2

    def __init__(self, nDeck=None):
        self.nDeck = nDeck

    def start(self):
        s = BlackjackState(self.nDeck)
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
