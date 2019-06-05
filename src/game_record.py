import joblib
import numpy as np
from pathlib import Path
import pachi_py

from model_config import ModelConfig


class GoGameRecord(object):
    persisted_fields = ('boards', 'player', 'moves', 'reward', 'values', 'action_distributions')
    """keep record of the moves from winner's perspective"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.moves = []  # move for current board, list of scalar
        self.boards = []  # list of np.ndarray (board_size, board_size), each element is encoded player
        self.values = []  # [optional] values  list of scalar
        self.action_distributions = []  # [optional] list of np.ndarray (action space size, )
        self.player = pachi_py.EMPTY
        self.reward = 0  # reward from black player's perspective

    def __len__(self):
        return len(self.moves)

    @classmethod
    def color_to_player(cls, color):
        if color is None:
            return pachi_py.EMPTY
        _map_color_to_player = {
            'b': pachi_py.BLACK,
            'w': pachi_py.WHITE,
        }
        assert color in _map_color_to_player
        return _map_color_to_player[color]

    @classmethod
    def player_to_color(cls, player):
        assert player in (pachi_py.WHITE, pachi_py.BLACK, pachi_py.EMPTY)
        return 'x' if player == pachi_py.BLACK else 'o' if player == pachi_py.WHITE else '.'

    @classmethod
    def encoded_board_to_array(cls, config, encoded):
        """
from encode_board
cdef enum Channel:
    CHAN_CELL_BLACK = 0
    CHAN_CELL_WHITE
    CHAN_CELL_EMPTY
cdef int _NUM_FEATURE_CHANNELS = 3
        """
        assert isinstance(encoded, np.ndarray), f'{type(encoded)}'
        assert encoded.ndim == 3, f'{encoded.shape}'
        assert encoded.shape[0] == 3, f'{encoded.shape}'
        assert encoded.shape[1] == config.board_size, f'{encoded.shape}'
        assert encoded.shape[2] == config.board_size, f'{encoded.shape}'
        return (encoded[0] * pachi_py.BLACK + encoded[1] * pachi_py.WHITE).astype(np.int8)

    @classmethod
    def from_encoded_board(cls, config: ModelConfig, pair, player, reward):
        assert len(pair)
        game_record = GoGameRecord(config)
        game_record.player = player
        game_record.reward = reward
        for encoded, action in pair:
            assert action < config.action_space_size
            game_record.boards.append(cls.encoded_board_to_array(config, encoded))
            game_record.moves.append(action)
        return game_record

    @classmethod
    def from_file(cls, config, in_file):
        assert Path(in_file).exists(), in_file
        game_record = GoGameRecord(config)
        in_dict = joblib.load(in_file)
        for k in cls.persisted_fields:
            assert k in in_dict, f'{k} not in {in_file}'
            setattr(game_record, k, in_dict[k])
        assert len(game_record.boards) == len(game_record.moves)
        return game_record

    def write_to_path(self, out_path):
        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True, parents=True)
        num_record = len(list(out_path.glob('*.joblib')))
        out_file = out_path/f'game_record-{num_record}.joblib'
        self.write_to_file(out_file)

    def write_to_file(self, out_file):
        out_dict = {k: getattr(self, k) for k in self.persisted_fields}
        joblib.dump(out_dict, out_file)
        read_back = joblib.load(out_file)
        assert len(read_back['moves']) == len(self.moves)
        assert len(read_back['boards']) == len(self.boards)
        assert read_back['player'] == self.player

    def render_result(self):
        print(f'player={self.player_to_color(self.player)}, reward={self.reward}, total moves={len(self)}')
        for i in range(len(self.boards)):
            for row in self.boards[i]:
                print(' '.join([self.player_to_color(p) for p in row]))
            print(f'{i}, action={self.moves[i]}')

    def action(self, ply_index):
        assert ply_index < len(self)
        if ply_index < len(self.action_distributions):
            # distribution is present
            return self.action_distributions[ply_index]
        action_distribution = np.zeros(self.config.action_space_size, dtype=float)
        action_distribution[self.moves[ply_index]] = 1.0
        return action_distribution


