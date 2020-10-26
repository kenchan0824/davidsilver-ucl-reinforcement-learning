import numpy as np

class BlackjackEnv:

    STATE_DIM = (22, 12, 2)
    ACTION_DIM = 2


    def __init__(self):

        self._deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        self._player = None
        self._dealer = None
        self.done = False


    def observe(self):
        
        return (self._sumHand(self._player),
                self._sumHand(self._dealer),
                int(self._usableAce(self._player)))


    def reset(self, player=None, dealer=None):

        self._player = [self._drawCard()] if player is None else player
        self._dealer = [self._drawCard()] if dealer is None else dealer
        self.done  = False
        return self.observe()


    def step(self, action):
        
        if self.done:
            return self.observe(), 0, True

        # player's turn
        reward = 0
        if (action == 0):                                   # hit
            card = self._drawCard()
            self._player.append(card)
            playerSum = self._sumHand(self._player)
            if playerSum > 21:
                reward = -1
                self.done = True
            
        else:                                               # stick
            
            # dealer's turn
            playerSum = self._sumHand(self._player)
            dealerSum = self._sumHand(self._dealer)
            while dealerSum < playerSum and dealerSum <= 21:
                card = self._drawCard()
                self._dealer.append(card)
                dealerSum = self._sumHand(self._dealer)
            
            self.done = True
            if dealerSum > 21:
                reward = 1
            else:
                reward = -1
        
        return self.observe(), reward, self.done


    def _usableAce(self, hand):

        if (1 in hand) and (sum(hand) < 12):
            return True

        return False


    def _sumHand(self, hand):

        sumHand = sum(hand)
        if self._usableAce(hand):
            sumHand += 10

        return sumHand


    def _drawCard(self):

        card = np.random.choice(self._deck)
        return card
