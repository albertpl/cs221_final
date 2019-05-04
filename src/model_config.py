def add_to_parser(config_obj, parser):
    for k in config_obj.next_key():
        attr_type = type(getattr(config_obj, k))
        parser.add_argument('--'+k, dest=k, type=attr_type)
    return


class ModelConfig(object):
    # GO game configurations
    board_size = 19
    game_index_file = 'game_index.yaml'
    tree_depth = 8
    feature_channel = 17  # 2 * 8 + 1
    game_record_row_index = -3
    game_record_col_index = -2
    game_record_player_index = -1
    player_policy = ''
    max_time_step_per_episode = 1000
    # hyper parameters for model
    dense_layer_dim = 256
    batch_size = 64
    batch_size_inference = 64
    # hyper parameters for training
    training_epochs = 500
    save_weight_on_best_train = True
    lr_scheduler = 'clr_exp_range'
    lr_reset_every_epochs = 20
    lr_range = [1.0e-4, 3.0e-3]
    lr_decay = 0.99
    dropout_keep_prob = 1.0
    early_stop = 100000
    use_augmentation = False
    model_name = 'SL'
    print_n_per_epoch = 1000
    # parameters to be set during training
    iterations_per_epoch = 1
    learner_log_dir = '/tmp/log_dir/'
    weight_root = '/tmp/weights/'
    allow_weight_init = True
    dataset_path = '/tmp/go_games/'

    def __init__(self, **kwargs):
        if kwargs is not None:
            self.update_key(**kwargs)

    def __repr__(self):
        return self.to_string(verbose=False)

    def to_string(self, verbose=False, delimit='___'):
        s = ''
        for key in self.next_key():
            s += key + '=' + str(getattr(self, key)) + delimit
        return s

    def next_key(self):
        for key in sorted([a for a in dir(self)
                           if not callable(getattr(self, a)) and not a.startswith("__")]):
            yield key

    def update_key(self, **kwargs):
        for key in self.next_key():
            if key in kwargs and kwargs[key] is not None:
                attr_type = type(getattr(self, key))
                v = attr_type(kwargs[key])
                setattr(self, key, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.next_key()}
