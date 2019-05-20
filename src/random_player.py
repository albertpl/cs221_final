from environment import GoState
from player import Player


class RandomPlayer(Player):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def next_action(self, curr_state: GoState, prev_state, prev_action):
        return curr_state.random_action()

