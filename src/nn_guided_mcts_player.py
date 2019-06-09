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
        return self.update_statistics()

    def mc_update(self, node: SearchTreeNode, path: []):
        assert -1.0 <= node.estimated_value <= 1.0, node.estimated_value
        self.rollout_boards = []
        reward = -node.estimated_value
        for node, action in path[::-1]:
            node.visit_counts[action] += 1
            node.sum_action_values[action] += reward
            reward *= -1.0

    def update_node(self, node: SearchTreeNode):
        if node.state.board.is_terminal:
            black_win = GoEnv.game_result(self.config, node.state)[0]
            node.estimated_value = 1.0 if black_win else -1.0
        # otherwise run inference to update priors and values
        if not node.is_root:
            encoded = node.state.board.encode()
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
        # first batch of the first result
        batch_output = self.model_controller.infer(batch_input)
        node.estimated_value = batch_output.result[1][0]
        if self.config.mcts_set_value_for_q:
            node.sum_action_values = np.ones_like(node.sum_action_values) * node.estimated_value
        priors = np.where(node.prior_probabilities > 0.0, batch_output.result[0][0], 0.0)
        if self.config.print_board > 1 and node.is_root:
            print(batch_output.result[0][0])
            print(node.prior_probabilities)
            print(priors)
        if self.config.mcts_dirichlet_alpha > 0 and node.is_root:
            priors[node.legal_actions] = \
                0.75 * priors[node.legal_actions] + \
                0.25 * np.random.dirichlet([self.config.mcts_dirichlet_alpha] * len(node.legal_actions))
            if self.config.print_board > 1:
                print(priors)
        priors /= np.sum(priors+1e-8)
        node.prior_probabilities = priors

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


