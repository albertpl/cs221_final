import logging
import numpy as np
from pathlib import Path

from batch import BatchInput
from environment import GoState, GoEnv, resign_action, pass_action
import feature
from game_record import GoGameRecord
from model_config import ModelConfig
from model_controller import KerasModelController
from player import Player


class PolicyPlayer(Player):
    def __init__(self, config: ModelConfig, player, record_path=''):
        super().__init__(config=config, player=player, record_path=record_path)
        assert config.weight_root, f'please supply config.weight_root'
        config.allow_weight_init = True
        config.learner_log_dir = ''
        config.batch_size_inference = 1
        config.model_name = 'policy_gradient'
        self.model_controller = KerasModelController.controller(config)
        self.boards, self.actions, self.estimated_value = [], [], []
        self.model_controller.__enter__()

    def __del__(self):
        if hasattr(self, 'model_controller') and self.model_controller is not None:
            self.model_controller.__exit__(None, None, None)

    def reset(self, board):
        self.boards, self.actions, self.estimated_value = [], [], []

    def end_game(self, reward):
        if self.record_path and len(self.boards):
            assert len(self.boards) == len(self.actions)
            assert len(self.estimated_value) == len(self.actions)
            record = GoGameRecord(config=self.config)
            record.player = self.player
            record.reward = reward
            record.moves = self.actions
            record.boards = self.boards
            record.values = self.estimated_value
            record.write_to_path(self.record_path)

    def next_action(self, state: GoState, prev_state: GoState, prev_action):
        board_size = self.config.board_size
        ply_index = len(self.boards)
        encoded = state.board.encode()
        board = GoGameRecord.encoded_board_to_array(self.config, encoded=encoded)
        self.boards.append(board)
        feature_vector = feature.array_to_feature(self.config,
                                                  boards=self.boards,
                                                  player=state.color,
                                                  ply_index=ply_index)
        batch_input = BatchInput()
        batch_input.batch_xs = feature_vector[np.newaxis, ...]
        batch_output = self.model_controller.infer(batch_input)
        # first batch of the first result
        probabilities = batch_output.result[0][0]
        # return the most preferred legal action
        legal_actions = set(state.all_legal_actions())
        sampled_size = min(self.config.action_space_size, np.sum(probabilities > 0.0))
        action = pass_action(board_size)
        if sampled_size > 0:
            sampled_actions = np.random.choice(self.config.action_space_size,
                                               size=sampled_size,
                                               replace=False,
                                               p=probabilities)
            for sampled_action in sampled_actions:
                if sampled_action in legal_actions:
                    action = sampled_action
                    break
        else:
            print('no possible moves')
        self.actions.append(action)
        self.estimated_value.append(batch_output.result[1][0])
        return action





