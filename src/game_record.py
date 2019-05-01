import joblib
import numpy as np
from pathlib import Path
import pachi_py

from model_config import ModelConfig


class GoGameRecord(object):
    def __init__(self, config: ModelConfig, plays):
        assert isinstance(plays, np.ndarray)
        assert plays.ndim == 2, plays.shape
        assert plays.shape[0] > 0
        assert plays.shape[1] == self.record_size(config), plays.shape
        self.config = config
        self.plays = plays

    def __len__(self):
        return len(self.plays)

    @classmethod
    def record_size(cls, config: ModelConfig):
        # e.g., 19 x 19, row, col, player
        return config.board_size ** 2 + 3

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
    def from_file(cls, config, in_file):
        assert Path(in_file).exists(), in_file
        plays = joblib.load(in_file)
        return GoGameRecord(config, plays)

    def write_to_file(self, out_file):
        joblib.dump(self.plays, out_file)
        read_back = joblib.load(out_file)
        assert np.allclose(read_back, self.plays)

    def render_result(self):
        board_size = self.config.board_size
        last_play = self.plays[-1, :-3].reshape((board_size, board_size))
        for row in last_play:
            print(' '.join([self.player_to_color(p) for p in row]))
        print(f'action=({self.action(-1)}, player={self.player_to_color(self.player(-1))}')

    def action(self, ply_index):
        assert ply_index < len(self)
        board_size = self.config.board_size
        return int(self.plays[ply_index, self.config.game_record_row_index] * board_size +
                   self.plays[ply_index, self.config.game_record_col_index])

    def player(self, ply_index):
        assert ply_index < len(self)
        return self.plays[ply_index, self.config.game_record_player_index]

    @classmethod
    def opponent(cls, player):
        assert player in (pachi_py.WHITE, pachi_py.BLACK)
        return pachi_py.WHITE if player == pachi_py.BLACK else pachi_py.BLACK

    def feature(self, depth, ply_index):
        """
        return features from history of current index,
        -indicators for player, each position
        -indicators for opponent, each position
        -binary value for player, 1 for black, 0 for white
        :param depth:
        :return:
        array of shape
        (board_size, board_size, 2 * depth + 1) * board_size ** 2)
        """
        assert ply_index < len(self)
        assert self.config.feature_channel == 2 * self.config.tree_depth + 1, self.config
        board_size = self.config.board_size
        observation_size = board_size ** 2
        features = np.zeros((board_size, board_size, self.config.feature_channel))
        first_index = max(ply_index - depth, 0)
        player = self.player(ply_index)
        opponent = self.opponent(player)
        for i, index in enumerate(range(ply_index, first_index-1, -1)):
            features[:, :, i] = np.reshape(self.plays[index][:observation_size] == player,
                                           (board_size, board_size))
            features[:, :, depth + i] = np.reshape(self.plays[index][:observation_size] == opponent,
                                                   (board_size, board_size))
            features[:, :, -1] = player
        return features





