import numpy as np

from batch import BatchInput, BatchOutput
import environment
from game_record import GoGameRecord
from model_config import ModelConfig


def array_to_feature(config, boards, player, ply_index):
    """
    return features from history of current index,
    -indicators for player, each position
    -indicators for opponent, each position
    -binary value for player, 1 for black, 2 for white, compatible pachi_py.BLACK and pachi_py.WHITE
    :return:
    array of shape
    (board_size, board_size, 2 * depth + 1)
    """
    assert ply_index < len(boards)
    assert config.feature_channel == 2 * config.tree_depth + 1, config
    board_size, depth = config.board_size, config.tree_depth
    features = np.zeros((board_size, board_size, config.feature_channel), dtype=float)
    first_index = max(ply_index - depth, 0)
    opponent = environment.opponent(player)
    for i, index in enumerate(range(ply_index, first_index-1, -1)):
        features[:, :, i] = boards[index] == player
        features[:, :, depth + i] = boards[index] == opponent
        features[:, :, -1] = player
    return features


def create_batch_from_game_records(config: ModelConfig, game_records, ply_indices):
    """
    input: board array converted to features (see array_to_feature)
    policy target: greedy action in game record
    value target: reward
    """
    batch_size = len(ply_indices)
    batch = BatchInput()
    batch.batch_xs = np.zeros((batch_size, config.board_size, config.board_size, config.feature_channel))
    action_distribution = np.zeros((batch_size, config.action_space_size), dtype=float)
    value_target = np.zeros(batch_size, dtype=float)
    for b, (game_index, ply_index) in enumerate(ply_indices):
        assert game_index < len(game_records)
        game_record = game_records[game_index]
        assert isinstance(game_record, GoGameRecord)
        batch.batch_xs[b, ...] = array_to_feature(config=config,
                                                  boards=game_record.boards,
                                                  player=game_record.player,
                                                  ply_index=ply_index)
        action_distribution[b, game_record.action(ply_index=ply_index)] = 1.0
        value_target[b] = game_record.reward
    return batch.batch_xs, {'policy': action_distribution, 'value': value_target}


def create_batch_fn(config: ModelConfig):
    _model_to_fn = {
        'policy_with_baseline': create_batch_from_game_records,
    }
    assert config.model_name in _model_to_fn, config.model_name
    return _model_to_fn[config.model_name]

