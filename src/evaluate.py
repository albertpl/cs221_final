import argparse
import logging
import numpy as np
import time

from environment import GoEnv
from model_config import ModelConfig


def create_agent(config: ModelConfig, policy):
    if policy == 'random':
        from random_player import RandomPlayer
        return RandomPlayer(config)
    elif policy == 'nn':
        from nn_player import NetworkPlayer
        return NetworkPlayer(config)
    elif policy == 'pachi':
        from pachi_player import PachiPlayer
        return PachiPlayer(config)
    elif policy == 'mcts':
        from mcts_player import MCTSPlayer
        return MCTSPlayer(config)
    else:
        raise ValueError(f'unsupported policy = {config.player_policy}')


def evaluate():
    config = ModelConfig()
    config.print_board = args.print_board
    config.weight_root = args.weight_root
    config.game_record_path = args.record_path
    config.game_result_path = f'{args.result_path}/{args.black_policy}_vs_{args.white_policy}/'
    env = GoEnv(config,
                create_agent(config, args.black_policy),
                create_agent(config, args.white_policy))
    env.play(num_games=args.num_games)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    subparser = parser.add_subparsers(dest='command', title='sub_commands', description='valid commands')
    subparser.required = True

    sub_parser = subparser.add_parser('eval')
    sub_parser.add_argument('black_policy')
    sub_parser.add_argument('white_policy')
    sub_parser.add_argument('num_games', type=int)
    sub_parser.add_argument('--print_board', type=int, default=0)
    sub_parser.add_argument('--weight_root')
    sub_parser.add_argument('--record_path')
    sub_parser.add_argument('--result_path', default='/tmp/game_result')
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
