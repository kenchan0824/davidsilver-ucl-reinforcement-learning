import numpy as np

class Easy21Env:

    STATE_DIM = (22, 11)
    ACTION_DIM = 2


    def __init__(self):

        self._player = None
        self._dealer = None
        self.done = False


    def observe(self):

        return (sum(self._player), sum(self._dealer))


    def reset(self, player=None, dealer=None):

        deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cards = np.random.choice(deck, 2, replace=True)
        self._player = [cards[0]] if player is None else player
        self._dealer = [cards[1]] if dealer is None else dealer
        self.done = False
        return self.observe()


    def step(self, action):
        if self.done:
            return self.observe(), 0, True

        # player turn
        reward = 0
        if action == 0:                     # hit
            card = self._drawCard()
            self._player.append(card)
            playerSum = sum(self._player)
            if playerSum > 21 or playerSum < 1:
                self.done = True
                reward = -1

        else:                               # stick
            # dealer turn
            playerSum = sum(self._player)
            dealerSum = sum(self._dealer)            
            while dealerSum < 17 and dealerSum > 0:
                card = self._drawCard()
                self._dealer.append(card)
                dealerSum = sum(self._dealer)

            self.done = True
            if dealerSum > 21 or dealerSum < 1:
                reward = 1
            elif playerSum < dealerSum:
                reward = -1
            elif playerSum > dealerSum:
                reward = 1
            else:
                reward = 0

        return self.observe(), reward, self.done


    def _drawCard(self):
        deck = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*2 + [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
        card = np.random.choice(deck, replace=True)
        return card
