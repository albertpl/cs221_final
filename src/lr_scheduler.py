import numpy as np

from model_config import ModelConfig


class LRScheduler(object):
    def __init__(self, model_config):
        self.model_config = model_config

    def compute(self, num_iteration):
        raise NotImplemented


class CLRScheduler(LRScheduler):
    """
    https://arxiv.org/abs/1506.01186
    Cyclical Learning Rate with triangular cycle
    """
    def __init__(self, model_config):
        super().__init__(model_config)

    def compute(self, num_iteration):
        """
        :param: num_iteration
        :return: updated lr
        """
        model_config = self.model_config
        cycle = model_config.iterations_per_epoch * model_config.lr_reset_every_epochs
        policy = model_config.lr_scheduler
        base_lr = model_config.lr_range[0]
        max_lr = model_config.lr_range[-1]
        step_size = cycle/2
        gamma = np.float_power(model_config.lr_decay, 1.0/cycle)

        cycle = np.floor(1 + num_iteration / (2 * step_size))
        x = np.abs(num_iteration / step_size - 2 * cycle + 1)
        if policy == 'clr_triangular':
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
        elif policy == 'clr_exp_range':
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma ** num_iteration
        else:
            raise ValueError(f'unsupported policy={policy}')
        return lr


class SGDScheduler(LRScheduler):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.t_i = model_config.iterations_per_epoch * model_config.lr_reset_every_epochs
        assert not np.isclose(self.t_i, 0), f'lr_reset_every_epochs={model_config.lr_reset_every_epochs}'
        self.last_iter = 0
        # use fixed t_mult
        self.t_mult = 2

    def compute(self, num_iteration):
        """
        https://arxiv.org/abs/1608.03983
        STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
        :param: num_iterations: config object
        :return: updated lr
        """
        t_cur = num_iteration - self.last_iter
        if t_cur > self.t_i:
            self.last_iter += self.t_i
            t_cur = num_iteration - self.last_iter
            self.t_i *= 2
        eta_min = self.model_config.lr_range[0]
        eta_max = self.model_config.lr_range[-1]
        lr = eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * t_cur/self.t_i)) / 2
        return lr


class LearningRateSchedulerFacade(object):
    def __init__(self, model_config):
        assert isinstance(model_config, ModelConfig)
        self.model_config = model_config
        if model_config.lr_scheduler.startswith('clr'):
            self.scheduler = CLRScheduler(model_config)
        elif model_config.lr_scheduler == 'sgdr':
            self.scheduler = SGDScheduler(model_config)
        else:
            raise ValueError(f'unsupported lr scheduler algorithm={model_config.lr_scheduler}')

    def compute(self, num_iteration):
        return self.scheduler.compute(num_iteration)


