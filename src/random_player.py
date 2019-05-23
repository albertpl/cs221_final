from environment import GoState
import numpy as np
from player import Player


class RandomPlayer(Player):
    def __init__(self, config, player, record_path=''):
        super().__init__(config=config, player=player, record_path=record_path)

    def _next_action(self, curr_state: GoState, prev_state, prev_action):
        actions = curr_state.all_legal_actions()
        return np.random.choice(actions)

