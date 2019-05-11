from environment import coord_to_action
import numpy as np
from player import Player


class RandomPlayer(Player):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def next_action(self, curr_state, prev_state, prev_action):
        b = curr_state.board
        legal_coords = b.get_legal_coords(curr_state.color)
        chosen_action = coord_to_action(b, np.random.choice(legal_coords))
        return chosen_action




