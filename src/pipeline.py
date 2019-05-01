import argparse
import joblib
import logging
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

from model_config import ModelConfig
from model_controller import KerasModelController

logging_format = '%(asctime)s %(module)s:%(lineno)d %(funcName)s %(levelname)s %(message)s'


def train_sl():
    config = ModelConfig()
    dataset_root = Path(args.dataset_path)
    assert dataset_root.exists()
    config.dataset_path = args.dataset_path
    config.weight_root = args.weight_path
    config.learner_log_dir = args.log_dir
    model_controller = KerasModelController(config)
    model_controller.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    subparser = parser.add_subparsers(dest='command', title='sub_commands', description='valid commands')
    subparser.required = True

    sub_parser = subparser.add_parser('train_sl')
    sub_parser.add_argument('dataset_path', help='root path for dataset')
    sub_parser.add_argument('weight_path', help='path weight root, required if learning is bypassed')
    sub_parser.add_argument('--log_dir', help='path for monitor tool, e.g. tensorboard')
    sub_parser.set_defaults(func=train_sl)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format=logging_format)
    t0 = time.time()
    if args.command:
        args.func()
        logging.info("%s takes %f s" % (args.command, time.time()-t0))
    else:
        parser.print_help()
