import logging
import numpy as np
import pachi_py

from dataset import BatchInput
from environment import GoState, action_to_coord, resign_action
from model_config import ModelConfig
from model_controller import KerasModelController


class NetworkPlayer(object):
    def __init__(self, config: ModelConfig):
        self.config = config
        config.allow_weight_init = False
        config.learner_log_dir = ''
        config.batch_size_inference = 1
        self.model_controller = KerasModelController(config)
        self.player_boards = []
        self.opponent_boards = []
        self.model_controller.__enter__()

    def __del__(self):
        if self.model_controller is not None:
            self.model_controller.__exit__(None, None, None)

    def reset(self, board):
        self.player_boards = []
        self.opponent_boards = []

    def next_action(self, state: GoState, prev_state: GoState, prev_action):
        board_size = self.config.board_size
        depth = self.config.tree_depth
        observation = state.board.encode()
        _board_channels = [0, 1] if state.color == pachi_py.BLACK else [1, 0]
        self.player_boards = [observation[_board_channels[0]]] + self.player_boards
        self.opponent_boards = [observation[_board_channels[1]]] + self.opponent_boards
        assert len(self.player_boards) == len(self.opponent_boards)
        batch_input = BatchInput()
        batch_input.batch_xs = np.zeros((1, board_size, board_size, self.config.feature_channel))
        batch_input.batch_xs[0, ..., -1] = state.color
        for i in range(min(depth, len(self.player_boards))):
            batch_input.batch_xs[0, ..., i] = self.player_boards[i]
            batch_input.batch_xs[0, ..., i+depth] = self.opponent_boards[i]
        batch_output = self.model_controller.infer(batch_input)
        # first batch of the first result
        probabilities = batch_output.result[0][0]
        # return the most preferred legal action
        legal_coords = state.board.get_legal_coords(state.color)
        for action in np.argsort(probabilities)[::-1]:
            logging.debug(f'action={action}')
            logging.debug(f'action={action}, probability={probabilities[action]}')
            coord = action_to_coord(state.board, action)
            if coord in legal_coords:
                return action
        logging.debug(f'quit...')
        return resign_action(board_size)





