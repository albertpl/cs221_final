"""
Adapted from https://github.com/openai/gym/blob/6af4a5b9b2755606c4e0becfe1fc876d33130526/gym/envs/board_game/go.py
"""
import logging
import numpy as np
import pachi_py
import pandas as pd
from pathlib import Path
import sys
import six
from six import StringIO
import time
from tqdm import tqdm


from game_record import GoGameRecord
from model_config import ModelConfig
from player import Player

# The coordinate representation of Pachi (and pachi_py) is defined on a board
# with extra rows and columns on the margin of the board, so positions on the board
# are not numbers in [0, board_size**2) as one would expect. For this Go env, we instead
# use an action representation that does fall in this more natural range.


def pass_action(board_size):
    return board_size**2


def resign_action(board_size):
    return board_size**2 + 1


def coord_to_action(board, c):
    '''Converts Pachi coordinates to actions'''
    if c == pachi_py.PASS_COORD: return pass_action(board.size)
    if c == pachi_py.RESIGN_COORD: return resign_action(board.size)
    i, j = board.coord_to_ij(c)
    action = i*board.size + j
    return action


def action_to_coord(board, a):
    '''Converts actions to Pachi coordinates'''
    if a == pass_action(board.size): return pachi_py.PASS_COORD
    if a == resign_action(board.size): return pachi_py.RESIGN_COORD
    return board.ij_to_coord(a // board.size, a % board.size)


def str_to_action(board, s):
    return coord_to_action(board, board.str_to_coord(s.encode()))


class GoState(object):
    '''
    Go game state. Consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is different
    from Pachi's internal "coord_t" encoding.
    '''
    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in [pachi_py.BLACK, pachi_py.WHITE], 'Invalid player color'
        self.board, self.color = board, color

    def act(self, action):
        '''
        Executes an action for the current player

        Returns:
            a new GoState with the new board and the player switched
        '''
        # coordinate = action_to_coord(self.board, action)
        # legal_coords = self.board.get_legal_coords(self.color)
        # assert coordinate in legal_coords, f'player={self.color}, action={action}'
        return GoState(
            self.board.play(action_to_coord(self.board, action), self.color),
            pachi_py.stone_other(self.color))

    def clone(self):
        return GoState(self.board.clone(), self.color)

    def all_legal_actions(self):
        return [coord_to_action(self.board, c) for c in self.board.get_legal_coords(self.color)]

    def random_action(self):
        _, coord = self.board.play_random(self.color, return_action=True)
        return coord_to_action(self.board, coord)

    def __repr__(self):
        return 'To play: {}\n{}'.format(six.u(pachi_py.color_to_str(self.color)), self.board.__repr__().decode())


def make_pachi_policy(board, engine_type='uct', threads=1, pachi_timestr=''):
    engine = pachi_py.PyPachiEngine(board.clone(), engine_type, six.b('threads=%d' % threads))

    def pachi_policy(curr_state, prev_state, prev_action):
        if prev_state is not None:
            assert engine.curr_board == prev_state.board, \
                'Engine internal board is inconsistent with provided board. ' \
                'The Pachi engine must be called consistently as the game progresses.'
            prev_coord = action_to_coord(prev_state.board, prev_action)
            engine.notify(prev_coord, prev_state.color)
            engine.curr_board.play_inplace(prev_coord, prev_state.color)
        out_coord = engine.genmove(curr_state.color, pachi_timestr)
        out_action = coord_to_action(curr_state.board, out_coord)
        # print(f'pachi playing action = {out_action}, color={curr_state.color}, state={curr_state}')
        engine.curr_board.play_inplace(out_coord, curr_state.color)
        return out_action
    return pachi_policy


def _play(black_policy_fn, white_policy_fn, board_size=19):
    '''
    Samples a trajectory for two player policies.
    Args:
        black_policy_fn, white_policy_fn: functions that maps a GoState to a move coord (int)
    '''
    moves = []

    prev_state, prev_action = None, None
    curr_state = GoState(pachi_py.CreateBoard(board_size), pachi_py.BLACK)

    while not curr_state.board.is_terminal:
        a = (black_policy_fn if curr_state.color == pachi_py.BLACK else white_policy_fn)(curr_state, prev_state, prev_action)
        next_state = curr_state.act(a)
        moves.append((curr_state, a, next_state))

        prev_state, prev_action = curr_state, a
        curr_state = next_state

    return moves


class GoEnv(object):
    def __init__(self, config: ModelConfig, black_player: Player, white_player: Player):
        self.config = config
        self.board_size = config.board_size
        self.black_player = black_player
        self.white_player = white_player

    @classmethod
    def seed(cls, seed=None):
        pachi_py.pachi_srand(seed)
        return

    @classmethod
    def render(cls, state, mode="human"):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(state) + '\n')
        return outfile

    def play_game(self, curr_state, prev_state=None, prev_action=None, show_result=True):
        num_ply = 0
        black_pairs, white_pairs = [], []
        start_time = time.time()
        while not curr_state.board.is_terminal:
            if curr_state.color == pachi_py.BLACK:
                current_player = self.black_player
                pairs = black_pairs
            else:
                current_player = self.white_player
                pairs = white_pairs
            encoded_board = curr_state.board.encode() if self.config.game_record_path else None
            action = current_player.next_action(curr_state, prev_state, prev_action)
            pairs.append((encoded_board, action))
            try:
                next_state = curr_state.act(action)
                prev_state, prev_action = curr_state, action
                curr_state = next_state
                num_ply += 1
            except pachi_py.IllegalMove:
                six.reraise(*sys.exc_info())
        game_time = time.time() - start_time
        # We're in a terminal state. Reward is 1 if won, -1 if lost
        if not curr_state.board.is_terminal:
            print('game is out of time')
            self.render(curr_state)
        # komi is zero when the board is initialized
        # the official score is: komi + white score - black score + handicap (not used)
        score = self.config.komi + curr_state.board.official_score
        if score > 0:
            reward = -1.0
            pairs = white_pairs
            win_player = pachi_py.WHITE
        else:
            reward = 1.0
            pairs = black_pairs
            win_player = pachi_py.BLACK
        if self.config.game_record_path:
            game_record = GoGameRecord.from_encoded_board(self.config, pairs, win_player)
            game_record.write_to_path(self.config.game_record_path)
        if self.config.print_board > 0:
            self.render(curr_state)
        if show_result:
            logging.info(f'num_plys={num_ply}, reward={reward}, score={curr_state.board.official_score}')
        return {
            'reward': reward,
            'score': score,
            'plys': num_ply,
            'time': game_time,
        }

    def play(self, num_games=1):
        results = []
        for _ in tqdm(range(num_games)):
            curr_state = GoState(pachi_py.CreateBoard(self.board_size), pachi_py.BLACK)
            self.black_player.reset(curr_state.board)
            self.white_player.reset(curr_state.board)
            results.append(self.play_game(curr_state))
        results_pd = pd.DataFrame(results)
        if self.config.game_result_path:
            out_path = Path(self.config.game_result_path)
            out_path.mkdir(exist_ok=True, parents=True)
            num_result = len([out_path.glob('*.csv')])
            results_pd.to_csv(out_path/f'game_{num_result}.csv')
        rewards = results_pd['reward'].values
        logging.info(f'total games = {num_games}, black win rate = {np.sum(rewards>0)/num_games}')
        return

