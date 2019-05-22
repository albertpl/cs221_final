import logging
import math
import numpy as np
import pachi_py

from environment import GoState, GoEnv
from model_config import ModelConfig
from random_player import RandomPlayer
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
        self.illegal_cost = np.ones(config.action_space_size, dtype=float) * 1.0e5
        self.illegal_cost[np.array(state.all_legal_actions())] = 0.0
        # default to uniform
        self.prior_probabilities = np.ones(config.action_space_size, dtype=float)

    def next_action(self):
        """ UCB policy """
        action_values = self.sum_action_values/(self.visit_counts + 1e-8) - self.illegal_cost
        ucb = action_values \
            + self.c_puct * math.sqrt(np.sum(self.visit_counts)) * self.prior_probabilities/(1 + self.visit_counts)
        return int(np.argmax(ucb))


class MCTSPlayer(Player):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.root_node = None
        self.num_moves = 0

    def simulate(self, node):
        # default roll out policy: random
        env = GoEnv(self.config, black_player=RandomPlayer(self.config), white_player=RandomPlayer(self.config))
        return env.play_game(node.state)['reward']

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
                    successor_state = node.state.clone().act(action)
                    successor_node = SearchTreeNode(self.config, successor_state, parent=node)
                    node.children[action] = successor_node
                    logging.debug(f'expanding to successor=\n{successor_state}')
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
            action = np.random.choice(self.config.action_space_size, size=1, p=probabilities)
        else:
            action = np.argmax(self.root_node.visit_counts)
        # TODO: reuse root?
        self.root_node = None
        return action

    def reset(self, board):
        self.root_node = None
        self.num_moves = 0

    def next_action(self, state: GoState, prev_state: GoState, prev_action):
        self.num_moves += 1
        return self.search(state)



