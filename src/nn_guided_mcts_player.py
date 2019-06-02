import logging
import numpy as np


from batch import BatchInput
from environment import GoState, GoEnv
import feature
from game_record import GoGameRecord
from model_config import ModelConfig
from model_controller import KerasModelController
from mcts_player import MCTSPlayer, SearchTreeNode


class NNGuidedMCTSPlayer(MCTSPlayer):
    def __init__(self, config: ModelConfig, player, record_path):
        super().__init__(config=config, player=player, record_path=record_path)
        assert config.weight_root, f'please supply config.weight_root'
        config.allow_weight_init = True
        config.learner_log_dir = ''
        config.batch_size_inference = 1
        config.model_name = 'supervised'
        self.boards, self.action_distributions, self.actions = [], [], []
        self.rollout_boards = []  # used during roll out
        self.model_controller = KerasModelController.controller(config)
        self.model_controller.__enter__()

    def __del__(self):
        if hasattr(self, 'model_controller') and self.model_controller is not None:
            self.model_controller.__exit__(None, None, None)

    def reset(self, board):
        self.num_moves = 0
        self.boards, self.action_distributions, self.actions = [], [], []

    def end_game(self, reward):
        if self.record_path and len(self.boards):
            assert len(self.boards) == len(self.actions)
            record = GoGameRecord(config=self.config)
            record.player = self.player
            record.reward = reward
            record.moves = self.actions
            record.boards = self.boards
            record.action_distributions = self.action_distributions
            record.write_to_path(self.record_path)

    def simulate(self, node):
        assert -1.0 <= node.estimated_value <= 1.0, node.estimated_value
        return node.estimated_value

    def expand(self, node: SearchTreeNode, action):
        successor_state = node.state.clone().act(action)
        logging.debug(f'expanding to successor=\n{successor_state}')
        successor_node = SearchTreeNode(self.config, successor_state, parent=node)
        node.children[action] = successor_node
        if successor_state.board.is_terminal:
            successor_node.estimated_value = GoEnv.game_result(self.config, successor_state)
        else:
            encoded = successor_node.state.board.encode()
            board = GoGameRecord.encoded_board_to_array(self.config, encoded=encoded)
            self.rollout_boards.append(board)
            # perform inference to populate prior and evaluates
            boards = self.boards + self.rollout_boards
            feature_vector = feature.array_to_feature(self.config,
                                                      boards=boards,
                                                      player=node.state.color,
                                                      ply_index=(len(boards)-1))
            batch_input = BatchInput()
            batch_input.batch_xs = feature_vector[np.newaxis, ...]
            batch_output = self.model_controller.infer(batch_input)
            # first batch of the first result
            successor_node.prior_probabilities = batch_output.result[0][0]
            successor_node.estimated_value = batch_output.result[1][0]
        self.rollout_boards = []
        return successor_node

    def select(self, node: SearchTreeNode):
        action = node.next_action()
        successor_node = node.children[action]
        if successor_node:
            encoded = successor_node.state.board.encode()
            board = GoGameRecord.encoded_board_to_array(self.config, encoded=encoded)
            self.rollout_boards.append(board)
        return action

    def after_rollout(self, root_node: SearchTreeNode):
        probabilities = root_node.visit_counts/np.sum(root_node.visit_counts)
        if self.num_moves <= self.config.mcts_tao_threshold:
            action = np.random.choice(self.config.action_space_size, size=1, p=probabilities)[0]
        else:
            action = np.argmax(root_node.visit_counts)
        self.action_distributions.append(probabilities)
        self.actions.append(action)
        assert len(self.action_distributions) == len(self.boards)
        assert len(self.actions) == len(self.boards)
        return action

    def next_action(self, state: GoState, prev_state: GoState, prev_action):
        self.num_moves += 1
        encoded = state.board.encode()
        board = GoGameRecord.encoded_board_to_array(self.config, encoded=encoded)
        self.boards.append(board)
        return self.search(state)


