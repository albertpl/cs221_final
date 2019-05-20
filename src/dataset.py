import logging
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import time
import yaml

from game_record import GoGameRecord
from model_config import ModelConfig


class BatchInput(object):
    """ represents batch data """
    def __init__(self):
        self.batch_xs = None
        self.batch_ys = None


class BatchOutput(object):
    """represents batch output data"""
    def __init__(self):
        self.result = None


class Dataset(object):
    """represent the abstract dataset"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.generator = None
        self.batch_size = 0
        self.xs, self.ys = None, None
        self.game_records = []
        self.ply_indices = []
        return

    def __len__(self):
        return len(self.ply_indices)

    def shuffle_data(self):
        random.shuffle(self.ply_indices)

    def load_samples(self, ply_indices):
        """ load training samples from game record """
        config = self.config
        depth = config.tree_depth
        batch_size = len(ply_indices)
        batch = BatchInput()
        batch.batch_xs = np.zeros((batch_size, config.board_size, config.board_size, config.feature_channel))
        action_distribution = np.zeros((batch_size, config.action_space_size), dtype=float)
        value_target = np.zeros(batch_size, dtype=float)
        for b, (game_index, ply_index) in enumerate(ply_indices):
            assert game_index < len(self.game_records)
            game_record = self.game_records[game_index]
            assert game_record is not None
            batch.batch_xs[b, ...] = game_record.feature(depth=depth, ply_index=ply_index)
            action_distribution[b, game_record.action(ply_index=ply_index)] = 1.0
            value_target[b] = game_record.outcome
        return batch.batch_xs, {'policy': action_distribution, 'value': value_target}

    @classmethod
    def update_index_file(cls, config: ModelConfig, in_root: Path):
        index_file = in_root/config.game_index_file
        indices, game_record_files = [], set()
        num_current_game_records = 0
        t0 = time.time()
        if index_file.exists():
            with open(index_file, 'r') as in_fd:
                indices = yaml.load(in_fd)
                game_record_files = set(f for f, _ in indices)
                num_current_game_records = len(game_record_files)
        logging.info(f'searching game records in {in_root} to update index file')
        indices += [(str(game_record_file), len(GoGameRecord.from_file(config, game_record_file)))
                    for game_record_file in tqdm(in_root.glob('**/*.joblib'))
                    if str(game_record_file) not in game_record_files]
        if num_current_game_records != len(indices):
            with open(index_file, 'w') as out_fd:
                yaml.dump(indices, out_fd)
        assert len(indices) > 0, f'fail to update {index_file}'
        logging.info(f'updating {index_file} in {time.time() - t0}')
        return indices

    @classmethod
    def from_path(cls, config: ModelConfig, in_root):
        """
        load train/val/test datasets
        :param config:
        :param in_root:
        :return: DatasetContainer object
        """
        in_root = Path(in_root)
        assert in_root.exists() and in_root.is_dir(), in_root
        dataset = Dataset(config)
        indices = cls.update_index_file(config, in_root)
        t0 = time.time()
        if config.train_first_n_samples > 0:
            indices = indices[:config.train_first_n_samples]
        logging.info(f'loading {len(indices)} game records from {in_root}')
        for i, (file_name, ply_len) in tqdm(enumerate(indices)):
            game_record = GoGameRecord.from_file(config, file_name)
            assert game_record is not None, f'fail to load game record from {file_name}'
            dataset.game_records.append(game_record)
            dataset.ply_indices += [(i, j) for j in range(ply_len)]
        logging.info(f'loaded total games = {len(dataset.game_records)}, '
                     f'total ply = {len(dataset.ply_indices)}, '
                     f'in {time.time() - t0:.1f}s')
        return dataset

    def batch_generator(self, batch_size, random_sampling=True, augmentation=False, loop=False, return_raw=False):
        """
        :param augmentation:
        :param int batch_size:
        :param bool random_sampling:
        :param bool augmentation:
        :param bool loop:
        :param bool return_raw:
        :return: a BatchInput object or custom object (if custom batch processor is used)
        """
        assert batch_size <= len(self), f'batch_size={batch_size} < len(self)={len(self)}'
        while True:
            if random_sampling:
                self.shuffle_data()
            for start in range(0, len(self), batch_size):
                end = start + batch_size
                if end > len(self):
                    plys_prev = self.ply_indices[start:]
                    if random_sampling:
                        self.shuffle_data()
                    next_start = batch_size - len(plys_prev)
                    batch_plys = plys_prev + self.ply_indices[:next_start]
                else:
                    batch_plys = self.ply_indices[start:end]
                assert len(batch_plys) == batch_size, f'len(batch_plys)={len(batch_plys)} <> {batch_size}'
                yield self.load_samples(batch_plys)
            if not loop:
                break

    def next_batch(self, batch_size, loop=True, random_sampling=True, augmentation=False):
        """
        :param augmentation:
        :param int batch_size: must be smaller than len(self)
        :param bool loop: wrap around to boundary
        :param bool random_sampling:
        :param bool augmentation: apply data augmentation for each batch
        :return: BatchInput container object
        """
        assert batch_size <= self.max_batch_size(), f'batch_size={batch_size} > max={self.max_batch_size()}'
        self.batch_size = batch_size
        if self.generator is None:
            self.generator = self.batch_generator(batch_size, random_sampling, augmentation)
        is_done = False
        while not is_done:
            try:
                return next(self.generator)
            except StopIteration:
                if loop:
                    self.generator = self.batch_generator(batch_size, random_sampling, augmentation)
                else:
                    is_done = True
                    self.generator = None
        raise StopIteration

    def max_batch_size(self):
        return len(self)


class DatasetContainer(object):
    def __init__(self):
        self.train, self.val, self.test = None, None, None

    @classmethod
    def from_path(cls, config: ModelConfig, in_root):
        in_root = Path(in_root)
        assert in_root.exists() and in_root.is_dir(), in_root
        container = DatasetContainer()
        for sub_dir in ('train', 'val', 'test'):
            dataset_path = in_root/sub_dir
            if dataset_path.exists():
                setattr(container, sub_dir, Dataset.from_path(config, dataset_path))
        return container


