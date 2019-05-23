import argparse
import logging
import numpy as np
import pachi_py
import time

from environment import GoEnv
from model_config import ModelConfig


def create_agent(config: ModelConfig, policy, player):
    assert player in (pachi_py.BLACK, pachi_py.WHITE)
    record_path = config.black_player_record_path if player == pachi_py.BLACK else config.white_player_record_path
    if policy == 'random':
        from random_player import RandomPlayer
        return RandomPlayer(config, player=player, record_path=record_path)
    elif policy == 'nn':
        from policy_player import PolicyPlayer
        return PolicyPlayer(config, player=player, record_path=record_path)
    elif policy == 'pachi':
        from pachi_player import PachiPlayer
        return PachiPlayer(config, player=player, record_path=record_path)
    elif policy == 'mcts':
        from mcts_player import MCTSPlayer
        return MCTSPlayer(config, player=player, record_path=record_path)
    else:
        raise ValueError(f'unsupported policy = {config.player_policy}')


def evaluate():
    config = ModelConfig(**vars(args))
    if args.result_root:
        config.game_result_path = f'{args.result_root}/{args.black_policy}_vs_{args.white_policy}/'
    print(config)
    env = GoEnv(config,
                create_agent(config, args.black_policy, pachi_py.BLACK),
                create_agent(config, args.white_policy, pachi_py.WHITE))
    env.play(num_games=args.num_games)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    ModelConfig().add_to_parser(parser)
    subparser = parser.add_subparsers(dest='command', title='sub_commands', description='valid commands')
    subparser.required = True

    sub_parser = subparser.add_parser('eval')
    sub_parser.add_argument('black_policy')
    sub_parser.add_argument('white_policy')
    sub_parser.add_argument('num_games', type=int)
    sub_parser.add_argument('--result_root')
    sub_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    np.set_printoptions(precision=2, suppress=True, linewidth=300)
    logging_format = '%(asctime)s %(module)s:%(lineno)d %(funcName)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format=logging_format)
    start_time = time.time()
    if args.command:
        args.func()
        logging.info("%s takes %f s" % (args.command, time.time()-start_time))
    else:
        parser.print_help()
