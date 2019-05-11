import logging
import numpy as np
import pachi_py
import six

from environment import GoState, action_to_coord, coord_to_action
from model_config import ModelConfig
from player import Player


class PachiPlayer(Player):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.engine = None

    def reset(self, board):
        # necessary because a pachi engine is attached to a game via internal data in a board
        # so with a fresh game, we need a fresh engine
        self.engine = pachi_py.PyPachiEngine(board.clone(), six.b('uct'), six.b('threads=1'))

    def next_action(self, state: GoState, prev_state: GoState, prev_action):
        assert self.engine
        if prev_state is not None:
            assert self.engine.curr_board == prev_state.board, \
                'Engine internal board is inconsistent with provided board. ' \
                'The Pachi engine must be called consistently as the game progresses.'
            prev_coord = action_to_coord(prev_state.board, prev_action)
            self.engine.notify(prev_coord, prev_state.color)
            self.engine.curr_board.play_inplace(prev_coord, prev_state.color)
        out_coord = self.engine.genmove(state.color, six.b(self.config.pachi_timestr))
        out_action = coord_to_action(state.board, out_coord)
        # print(f'pachi playing action = {out_action}, color={state.color}, state={state}')
        self.engine.curr_board.play_inplace(out_coord, state.color)
        return out_action




