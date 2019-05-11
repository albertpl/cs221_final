import joblib
import numpy as np
from pathlib import Path
import pachi_py

from model_config import ModelConfig


class GoGameRecord(object):
    persisted_fields = ('boards', 'player', 'moves')
    """keep record of the moves from winner's perspective"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.moves = []  # move for current board
        self.boards = []  # list of np.ndarray (board_size, board_size), each element is encoded player
        self.player = pachi_py.EMPTY

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
    def from_encoded_board(cls, config: ModelConfig, pairs, win_player):
        assert len(pairs)
        game_record = GoGameRecord(config)
        game_record.player = win_player
        for encoded, action in pairs:
            assert isinstance(encoded, np.ndarray), f'{type(encoded)}'
            assert encoded.ndim == 3, f'{encoded.shape}'
            assert encoded.shape[0] == 3, f'{encoded.shape}'
            assert encoded.shape[1] == config.board_size, f'{encoded.shape}'
            assert encoded.shape[2] == config.board_size, f'{encoded.shape}'
            assert action < config.action_space_size
            board = (encoded[0] * pachi_py.BLACK + encoded[1] * pachi_py.WHITE).astype(np.int8)
            game_record.boards.append(board)
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
        for row in self.boards[-1]:
            print(' '.join([self.player_to_color(p) for p in row]))
        print(f'player={self.player_to_color(self.player)}, total moves={len(self)}')

    def action(self, ply_index):
        assert ply_index < len(self)
        return self.moves[-1]

    @property
    def outcome(self):
        # TODO: right now, we only record from wining player
        return 1.0

    def opponent(self):
        assert self.player in (pachi_py.BLACK, pachi_py.WHITE)
        return pachi_py.WHITE if self.player == pachi_py.BLACK else pachi_py.BLACK

    def feature(self, depth, ply_index):
        """
        return features from history of current index,
        -indicators for player, each position
        -indicators for opponent, each position
        -binary value for player, 1 for black, 2 for white, compatible pachi_py.BLACK and pachi_py.WHITE
        :param depth:
        :return:
        array of shape
        (board_size, board_size, 2 * depth + 1)
        """
        assert ply_index < len(self)
        assert self.config.feature_channel == 2 * self.config.tree_depth + 1, self.config
        board_size = self.config.board_size
        features = np.zeros((board_size, board_size, self.config.feature_channel), dtype=float)
        first_index = max(ply_index - depth, 0)
        opponent = self.opponent()
        for i, index in enumerate(range(ply_index, first_index-1, -1)):
            features[:, :, i] = self.boards[index] == self.player
            features[:, :, depth + i] = self.moves[index] == opponent
            features[:, :, -1] = self.player
        return features

