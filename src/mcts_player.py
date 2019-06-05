import logging
import numpy as np
import pachi_py
from tqdm import tqdm
import six
import sys

from environment import GoState, GoEnv, pass_action, resign_action
from model_config import ModelConfig
from player import Player


class SearchTreeNode(object):
    def __init__(self, config: ModelConfig, state: GoState, parent):
        self.state = state
        self.parent = parent
        # each edge (s, a) stores a prior, visit count and action value Q
        self.visit_counts = np.zeros(config.action_space_size, dtype=int)
        self.sum_action_values = np.zeros(config.action_space_size, dtype=float)
        self.children = [None] * config.action_space_size
        self.c_puct = config.mcts_c_puct
        legal_actions = state.all_legal_actions()
        # punish illegal actions
        self.illegal_cost = np.ones(config.action_space_size, dtype=float) * 1.0e6
        self.illegal_cost[legal_actions] = 0.0
        # default to uniform
        self.prior_probabilities = np.zeros(config.action_space_size, dtype=float)
        self.prior_probabilities[legal_actions] = 1.0/len(legal_actions)
        self.config = config
        self.estimated_value = 0.0
        self.legal_actions = legal_actions
        self.is_root = False

    def next_action(self):
        """ UCB policy """
        action_values = self.sum_action_values/(self.visit_counts + 1e-8) - self.illegal_cost
        ucb = action_values \
            + self.c_puct * np.sqrt(np.sum(self.visit_counts)) * self.prior_probabilities/(1 + self.visit_counts)
        action = int(np.argmax(ucb))
        assert action in self.legal_actions
        return action

    def debug(self, action):
        print(f'illegal action: {action}, cost={self.illegal_cost}')
        action_values = self.sum_action_values/(self.visit_counts + 1e-8) - self.illegal_cost
        print(f'action values = \n{action_values}')
        ucb = action_values + \
            self.c_puct * np.sqrt(np.sum(self.visit_counts)) * self.prior_probabilities/(1 + self.visit_counts)
        print(f'ucb = \n{ucb}')
        return


class MCTSPlayer(Player):
    def __init__(self, config: ModelConfig, player, record_path):
        super().__init__(config=config, player=player, record_path=record_path)
        self.num_moves = 0

    def simulate(self, node: SearchTreeNode):
        if node.state.board.is_terminal:
            return GoEnv.game_result(self.config, node.state)[0]
        # default roll out policy: random
        from random_player import RandomPlayer
        black_player = RandomPlayer(self.config, player=pachi_py.BLACK, record_path='')
        white_player = RandomPlayer(self.config, player=pachi_py.WHITE, record_path='')
        env = GoEnv(self.config, black_player=black_player, white_player=white_player)
        result, *_ = env.play_game(node.state)
        return result['black_win']

    def update_node(self, node):
        pass

    def expand(self, node: SearchTreeNode, action):
        successor_state = node.state.clone().act(action)
        logging.debug(f'expanding to successor=\n{successor_state}')
        successor_node = SearchTreeNode(self.config, successor_state, parent=node)
        node.children[action] = successor_node
        self.update_node(successor_node)
        return successor_node

    def select(self, node: SearchTreeNode):
        return node.next_action()

    def after_rollout(self, root_node: SearchTreeNode):
        logging.debug(root_node.sum_action_values)
        if self.num_moves <= self.config.mcts_tao_threshold:
            probabilities = root_node.visit_counts/np.sum(root_node.visit_counts)
            action = GoEnv.sample_action(self.config, root_node.state, probabilities)
        else:
            action = np.argmax(root_node.visit_counts)
        return action

    def search(self, state):
        root_state = state.clone()
        root_node = SearchTreeNode(self.config, state=root_state, parent=None)
        root_node.is_root = True
        for i in range(self.config.mcts_num_rollout):
            node = root_node
            path = []
            # for each roll out
            # 1. traverse to leaf node, recursively, following tree policy: UCB
            while not node.state.board.is_terminal:
                action = self.select(node)
                path.append((node, action))
                if node.children[action] is None:
                    # 2. expand the leaf with successor state
                    try:
                        node = self.expand(node=node, action=action)
                    except pachi_py.IllegalMove:
                        node.debug(action)
                        six.reraise(*sys.exc_info())
                    break
                node = node.children[action]
            # 3. carry out simulation
            black_win = self.simulate(node)
            # 4. back up
            black_reward = 1.0 if black_win else -1.0
            white_reward = 0 - black_reward
            for node, action in path:
                node.visit_counts[action] += 1
                player_reward = black_reward if node.state.color == pachi_py.BLACK else white_reward
                node.sum_action_values[action] += player_reward
        # select the action from root node statistics
        action = self.after_rollout(root_node)
        # TODO: reuse root?
        return action

    def reset(self, board):
        self.num_moves = 0

    def _next_action(self, state: GoState, prev_state: GoState, prev_action):
        self.num_moves += 1
        return self.search(state)




