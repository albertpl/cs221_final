import argparse
import copy
from hyperopt import fmin, tpe, rand, hp, Trials, pyll
from hyperopt import STATUS_OK, STATUS_FAIL
import logging
import numpy as np
from pathlib import Path
import pachi_py
import pprint
import time
from typing import Dict
import traceback
from tqdm import tqdm
import yaml

from environment import GoEnv
from evaluate import create_agent
from model_config import ModelConfig

_trial_out_file = 'hpo.yaml'
pp = pprint.PrettyPrinter(indent=4, width=100)


def duplicate_and_update(origin: ModelConfig, space: Dict):
    duplicated = copy.deepcopy(origin)
    duplicated.update_key(**space)
    for k, v in space.items():
        if isinstance(v, dict):
            duplicated.update_key(**v)
    print(duplicated)
    return duplicated


def create_work_dir(config: ModelConfig):
    work_root = Path(config.hpo_work_root)
    trial_id = len(list(work_root.glob('trial*')))
    work_dir = work_root/f'trial-{trial_id}'
    work_dir.mkdir()
    return trial_id, work_dir


def evaluate(config: ModelConfig, space: Dict, num_games: int):
    duplicated = duplicate_and_update(config, space)
    trial_id, work_dir = create_work_dir(duplicated)
    config.game_result_path = work_dir
    try:
        env = GoEnv(config,
                    create_agent(config, config.black_player_policy, pachi_py.BLACK),
                    create_agent(config, config.white_player_policy, pachi_py.WHITE))
        results_pd = env.play(num_games=num_games)
        wins = results_pd['black_win'].values
        ret = {'loss': 1.0 - np.mean(wins), 'status': STATUS_OK}
    except (ValueError, AssertionError, ModuleNotFoundError, RuntimeError) as e:
        tb = traceback.format_exc(limit=None)
        logging.error(f'hpo exception: \ntb=\n{tb}\nhpo space=\n{pp.pformat(space)}')
        ret = {'loss': 1.0, 'status': STATUS_FAIL}
    result_out_file = work_dir/_trial_out_file
    duplicated.write_to_file(work_dir/'config.yaml')
    with result_out_file.open(mode='w') as out_fd:
        yaml.dump({
            'hpo_results': ret,
            'hpo_space': space,
            'work-dir': str(work_dir),
        }, out_fd)
    return ret


def search_space_mcts():
    return {
        'mcts_c_puct': hp.choice('mcts_c_puct', [0.2]),
        'mcts_num_rollout': hp.choice('mcts_num_rollout', [1000]),
        'mcts_tao_threshold': hp.choice('mcts_tao_threshold', [7]),
        'mcts_dirichlet_alpha': hp.choice('mcts_dirichlet_alpha', [0.0, 0.1, 0.5, 1.0, 10.0]),
    }


def mcts():
    mode = 'mcts'
    algo = tpe.suggest

    config = ModelConfig()
    config.hpo_work_root = args.work_root
    config.hpo_max_trial = args.max_trial
    config.black_player_policy = 'mcts'
    config.white_player_policy = 'pachi'
    _map_from_mode = {
        'mcts': (search_space_mcts(), lambda x: evaluate(config=config, space=x, num_games=args.num_games)),
    }
    assert mode in _map_from_mode
    space, objective_fn = _map_from_mode[mode]

    if args.show_space:
        for _ in range(10):
            pp.pprint(pyll.stochastic.sample(space))
        return

    trials = Trials()
    best = fmin(fn=objective_fn,
                space=space,
                algo=algo,
                max_evals=config.hpo_max_trial,
                trials=trials)
    pp.pprint(f'best={best}')
    candidates = [t for t in trials.trials if t['result']['status'] == STATUS_OK]
    if len(candidates) >= args.best_n:
        candidates.sort(key=lambda x: float(x['result']['loss']))
        for trial in candidates[:args.best_n]:
            print(f"loss={trial['result']['loss']}, space={pp.pformat(trial['misc'])}")


def show_top_results():
    in_root = Path(args.in_root)
    assert in_root.exists(), in_root
    results = [yaml.load(in_f.open()) for in_f in tqdm(in_root.glob(f'trial-*/{_trial_out_file}'))]
    results = [x for x in results if x['hpo_results']['status'] == STATUS_OK]
    results.sort(key=lambda x: x['hpo_results']['loss'])
    print(f'total trials = {len(results)}, top_n={args.top_n}')
    for r in results[:args.top_n]:
        pp.pprint(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    ModelConfig().add_to_parser(parser)
    subparser = parser.add_subparsers(dest='command', title='sub_commands', description='valid commands')
    subparser.required = True

    sub_parser = subparser.add_parser('mcts')
    sub_parser.add_argument('work_root')
    sub_parser.add_argument('--show_space', action='store_true')
    sub_parser.add_argument('--best_n', default=100, type=int)
    sub_parser.add_argument('--num_games', default=100, type=int)
    sub_parser.add_argument('--max_trial', default=1, type=int)
    sub_parser.set_defaults(func=mcts)

    sub_parser = subparser.add_parser('show')
    sub_parser.add_argument('in_root')
    sub_parser.add_argument('--top_n', default=100, type=int)
    sub_parser.set_defaults(func=show_top_results)

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
