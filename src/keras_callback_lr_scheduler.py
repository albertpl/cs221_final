from keras.callbacks import Callback
from keras import backend as K

from lr_scheduler import LearningRateSchedulerFacade
from model_config import ModelConfig


class KerasCBLRScheduler(Callback):
    def __init__(self, config):
        super().__init__()
        assert isinstance(config, ModelConfig)
        self.config = config
        self.num_iteration = 0
        self.lrs_obj = LearningRateSchedulerFacade(config)
        return

    def on_train_begin(self, logs=None):
        self.num_iteration = 0

    def on_batch_begin(self, batch, logs=None):
        new_lr = self.lrs_obj.compute(self.num_iteration)
        K.set_value(self.model.optimizer.lr, new_lr)
        if logs:
            logs['lr'] = new_lr
        self.num_iteration += 1
