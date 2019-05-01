from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from pathlib import Path
import tensorflow as tf

from model_config import ModelConfig


class KerasCBSummaryWriter(Callback):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_iteration = 0  # increase across training
        self.print_every = max(
            config.iterations_per_epoch//config.print_n_per_epoch,
            1)

        model_name = self.config.model_name
        path_log_root = Path(config.learner_log_dir)
        path_log_root.mkdir(exist_ok=True, parents=True)
        n_log = len([p for p in path_log_root.iterdir() if p.is_dir() and p.match(model_name+'*')])
        path_log_dir = path_log_root/f'{model_name}_{n_log+1}'
        print(f'Tensorboard in {path_log_dir}')
        self.writer = tf.summary.FileWriter(str(path_log_dir), K.get_session().graph)
        return

    def on_train_begin(self, logs=None):
        self.num_iteration = 0

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = f'{self.config.model_name}/{name}'
            self.writer.add_summary(summary, index)
        self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        # batch reset to zero on each epoch
        self.num_iteration += 1
        if (self.num_iteration + 1) % self.print_every == 0:
            self._write_logs(logs, self.num_iteration)

    def on_epoch_end(self, epoch, logs=None):
        # different metrics are available than on_batch_end
        self._write_logs(logs, self.num_iteration)

