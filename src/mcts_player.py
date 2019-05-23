import logging
import math
import numpy as np
import pachi_py
from tqdm import tqdm
import six
import sys

from environment import GoState, GoEnv, pass_action, resign_action
from model_config import ModelConfig
from player import Player


class SearchTreeNode(object):
    def __init__(self, config: ModelConfig, state, parent):
        self.state = state
        self.parent = parent
        # each edge (s, a) stores a prior, visit count and action value Q
        self.visit_counts = np.zeros(config.action_space_size, dtype=int)
        self.sum_action_values = np.zeros(config.action_space_size, dtype=float)
        self.children = [None] * config.action_space_size
        self.c_puct = config.mcts_c_puct
        # punish illegal actions
        self.illegal_cost = np.ones(config.action_space_size, dtype=float) * 1.0e6
        legal_actions = state.all_legal_actions()
        self.illegal_cost[np.array(legal_actions)] = 0.0
        # default to uniform
        self.prior_probabilities = np.ones(config.action_space_size, dtype=float)
        self.config = config

    def next_action(self):
        """ UCB policy """
        action_values = self.sum_action_values/(self.visit_counts + 1e-8) - self.illegal_cost
        ucb = action_values \
            + self.c_puct * math.sqrt(np.sum(self.visit_counts)) * self.prior_probabilities/(1 + self.visit_counts)
        action = int(np.argmax(ucb))
        return action

    def debug(self, action):
        print(f'illegal action: {action}, cost={self.illegal_cost}')
        action_values = self.sum_action_values/(self.visit_counts + 1e-8) - self.illegal_cost
        print(f'action values = \n{action_values}')
        ucb = action_values + \
            self.c_puct * math.sqrt(np.sum(self.visit_counts)) * self.prior_probabilities/(1 + self.visit_counts)
        print(f'ucb = \n{ucb}')
        return


class MCTSPlayer(Player):
    def __init__(self, config: ModelConfig, player, record_path):
        super().__init__(config=config, player=player, record_path=record_path)
        self.root_node = None
        self.num_moves = 0

    def simulate(self, node):
        if self.config.mcts_simulation_policy == 'pachi':
            # for testing only
            from pachi_player import PachiPlayer
            black_player, white_player = PachiPlayer(self.config), PachiPlayer(self.config)
            black_player.reset(node.state.board.clone())
            white_player.reset(node.state.board.clone())
        else:
            # default roll out policy: random
            from random_player import RandomPlayer
            black_player = RandomPlayer(self.config, player=pachi_py.BLACK, record_path='')
            white_player = RandomPlayer(self.config, player=pachi_py.WHITE, record_path='')
        env = GoEnv(self.config, black_player=black_player, white_player=white_player)
        result, *_ = env.play_game(node.state)
        return result['reward']

    def search(self, state):
        root_state = state.clone()
        if self.root_node is None:
            self.root_node = SearchTreeNode(self.config, state=root_state, parent=None)
        for i in range(self.config.mcts_num_rollout):
            node = self.root_node
            path = []
            # for each roll out
            # 1. traverse to leaf node, recursively, following tree policy: UCB
            while True:
                action = node.next_action()
                path.append((node, action))
                logging.debug(f'[{i}] action={action}')
                if node.children[action] is None:
                    # 2. expand the leaf with successor state
                    try:
                        successor_state = node.state.clone().act(action)
                        successor_node = SearchTreeNode(self.config, successor_state, parent=node)
                        node.children[action] = successor_node
                        logging.debug(f'expanding to successor=\n{successor_state}')
                    except pachi_py.IllegalMove:
                        node.debug(action)
                        six.reraise(*sys.exc_info())
                    break
                node = node.children[action]
            # 3. carry out simulation
            reward = self.simulate(node)
            # 4. back up
            for node, action in path:
                node.visit_counts[action] += 1
                player_reward = reward if node.state.color == pachi_py.BLACK else (reward * (-1.0))
                node.sum_action_values[action] += player_reward
        # select the action from root node statistics
        logging.debug(self.root_node.sum_action_values)
        if self.num_moves <= self.config.mcts_tao_threshold:
            probabilities = self.root_node.visit_counts/np.sum(self.root_node.visit_counts)
            action = np.random.choice(self.config.action_space_size, size=1, p=probabilities)[0]
        else:
            action = np.argmax(self.root_node.visit_counts)
        # TODO: reuse root?
        self.root_node = None
        return action

    def reset(self, board):
        self.root_node = None
        self.num_moves = 0

    def _next_action(self, state: GoState, prev_state: GoState, prev_action):
        self.num_moves += 1
        return self.search(state)




