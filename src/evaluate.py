import argparse
import logging
import numpy as np
import pachi_py
import pandas as pd
import time
from tqdm import tqdm

from environment import GoEnv, make_random_policy, make_pachi_policy
from model_config import ModelConfig
from player import Player


class RandomPlayer(object):
    def __init__(self, env):
        self.policy = make_random_policy(env.np_random)

    def play(self, state):
        return self.policy(state)


def create_agent(config: ModelConfig, env):
    if config.player_policy == 'random':
        return RandomPlayer(env)
    elif config.player_policy == 'baseline':
        return Player(config)
    else:
        raise ValueError(f'unsupported policy = {config.player_policy}')


def evaluate_policy(config: ModelConfig, num_episode, print_board):
    env = GoEnv(player_color='black', illegal_move_mode='raise', board_size=19)
    total_rewards, total_steps = [], []
    t0 = time.time()
    for i_episode in range(num_episode):
        total_reward = 0.0
        encoded_board = env.reset()
        assert env.state.color == env.player_color
        agent = create_agent(config, env)
        t = 0
        for t in tqdm(range(config.max_time_step_per_episode)):
            if print_board > 1:
                env.render()
                logging.debug(encoded_board[:2, ...])
            action = agent.play(env.state)
            encoded_board, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                logging.debug(f"[{i_episode}]: t={t+1}, total_reward={total_reward}")
                if print_board > 0:
                    env.render()
                break
        total_rewards.append(total_reward)
        total_steps.append(t)
        logging.info(f'[{i_episode}]: reward={total_reward}, step={t}')
    env.close()
    t1 = time.time()
    rewards = pd.DataFrame(total_rewards)
    steps = pd.DataFrame(total_steps)
    print(f'total episode: {num_episode}, '
          f'\naverage reward: {rewards.describe()}, '
          f'\naverage time step: {steps.describe()}, '
          f'\ntime: {(t1-t0)/num_episode:.1f}s per game, ')
    return total_rewards


def evaluate():
    num_episode = args.num_episode
    config = ModelConfig()
    config.player = pachi_py.BLACK
    evaluate_policy(config, num_episode=num_episode, print_board=args.print_board)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    subparser = parser.add_subparsers(dest='command', title='sub_commands', description='valid commands')
    subparser.required = True

    sub_parser = subparser.add_parser('eval')
    sub_parser.add_argument('--num_episode', type=int, default=1)
    sub_parser.add_argument('--print_board', type=int, default=0)
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
