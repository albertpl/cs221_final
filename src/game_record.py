import joblib
import numpy as np
from pathlib import Path


class GoGameRecord(object):
    def __init__(self, plays):
        self.plays = plays

    @classmethod
    def from_file(cls, in_file):
        assert Path(in_file).exists(), in_file
        plays = joblib.load(in_file)
        return GoGameRecord(plays)

    def write_to_file(self, out_file):
        joblib.dump(self.plays, out_file)
        read_back = joblib.load(out_file)
        assert np.allclose(read_back, self.plays)

    def render_result(self):

        def p_to_c(v):
            return '.' if v == 0 else 'x' if v < 0 else 'o'
        board_size = int(np.sqrt(self.plays.shape[1] - 3))
        assert board_size == 19
        last_play = self.plays[-1, :-3].reshape((board_size, board_size))
        for row in last_play:
            print(' '.join([p_to_c(c) for c in row]))
        print(f'action=({self.plays[-1, -3]}, {self.plays[-1, -2]}), '
              f'color={p_to_c(self.plays[-1, -1])}')




