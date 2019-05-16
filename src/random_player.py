from environment import coord_to_action
import numpy as np
from player import Player


class RandomPlayer(Player):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def next_action(self, curr_state, prev_state, prev_action):
        return np.random.choice(curr_state.all_legal_actions())

