from model_config import ModelConfig
from game_record import GoGameRecord


class Player(object):
    def __init__(self, config: ModelConfig, player, record_path=''):
        self.config = config
        self.pair = []
        self.record_path = record_path
        self.player = player

    def reset(self, board):
        self.pair = []

    def end_game(self, reward):
        if self.record_path and len(self.pair):
            record = GoGameRecord.from_encoded_board(config=self.config,
                                                     pair=self.pair,
                                                     player=self.player,
                                                     reward=reward)
            record.write_to_path(self.record_path)

    def _next_action(self, state, prev_state, prev_action):
        raise NotImplemented

    def next_action(self, state, prev_state, prev_action):
        assert self.player == state.color
        action = self._next_action(state, prev_state=prev_state, prev_action=prev_action)
        if self.record_path:
            encoded_board = state.board.encode()
            self.pair.append((encoded_board, action))
        return action



