def add_to_parser(config_obj, parser):
    for k in config_obj.next_key():
        attr_type = type(getattr(config_obj, k))
        parser.add_argument('--'+k, dest=k, type=attr_type)
    return


class ModelConfig(object):
    seed = 23
    # GO game configurations
    komi = 5.5
    board_size = 9
    action_space_size = board_size * board_size + 2
    game_index_file = 'game_index.yaml'
    tree_depth = 8
    feature_channel = 17  # 2 * tree_depth + 1
    player_policy = ''
    max_time_step_per_episode = 1000
    print_board = 0
    # patchi
    pachi_timestr = '_2400'
    # parameters for MCTS
    mcts_c_puct = 1.0
    mcts_num_rollout = 100
    mcts_tao_threshold = 5  # following AGZ, before this threshold, tao=1, after, infinitesimal
    # parameters for NN model
    fc1_dim = 256
    fc2_dim = 128
    batch_size = 32
    batch_size_inference = 64
    # hyper parameters for training
    train_first_n_samples = 0
    training_epochs = 100
    save_weight_on_best_train = True
    lr_scheduler = 'clr_exp_range'
    lr_reset_every_epochs = 10
    lr_range = [1.0e-4, 1.0e-3]
    lr_decay = 0.99
    dropout_keep_prob = 1.0
    early_stop = 100000
    use_augmentation = False
    model_name = ''
    print_n_per_epoch = 1000
    iterations_per_epoch = 1
    allow_weight_init = True
    # paths
    game_record_path = ''
    game_result_path = '/tmp/game_result'
    learner_log_dir = '/tmp/log_dir/'
    weight_root = '/tmp/weights/'
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
