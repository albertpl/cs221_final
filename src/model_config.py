from pathlib import Path
import yaml


class ModelConfig(object):
    seed = 23
    # GO game configurations
    komi = 0.5
    board_size = 9
    action_space_size = board_size * board_size + 2  # pass and resign
    game_index_file = 'game_index.yaml'
    tree_depth = 8
    feature_channel = 17  # 2 * tree_depth + 1
    model_name = ''
    max_time_step_per_episode = 10000
    print_board = 0
    black_player_record_path = ''
    white_player_record_path = ''
    game_result_path = ''
    game_record_policy = ''
    black_player_policy = ''
    white_player_policy = ''
    # patchi
    pachi_timestr = '_2400'
    # parameters for MCTS
    mcts_c_puct = 2.5
    mcts_num_rollout = 1000
    mcts_tao_threshold = 7  # following AGZ, before this threshold, tao=1, after, infinitesimal
    mcts_simulation_policy = ''  # default is random
    mcts_dirichlet_alpha = 0.0  # 0.134 = 0.03 * 19^2 / 9 ^2
    mcts_set_value_for_q = 1
    pg_noise_threshold = 20
    pg_noise_alpha = 0.1
    # parameters for NN model
    fc1_dim = 128
    fc2_dim = 64
    batch_size = 32
    batch_size_inference = 32
    min_probability = 1e-8
    # hyper parameters for training
    train_first_n_samples = 0
    training_epochs = 10
    save_weight_on_best_train = True
    lr_scheduler = 'clr_exp_range'
    lr_reset_every_epochs = 10
    lr_range = [1.0e-4, 1.0e-3]
    lr_decay = 0.99
    dropout_keep_prob = 1.0
    early_stop = 100000
    use_augmentation = False
    print_n_per_epoch = 1000
    iterations_per_epoch = 1
    allow_weight_init = True
    learner_log_dir = ''
    weight_root = ''
    dataset_path = ''
    # HPO
    hpo_work_root = ''
    hpo_max_trial = 1

    def __init__(self, **kwargs):
        if kwargs is not None:
            self.update_key(**kwargs)

    def __repr__(self):
        return self.to_string(verbose=False)

    @classmethod
    def from_yaml(cls, yaml_file):
        assert Path(yaml_file).is_file(), f'{yaml_file}'
        with open(yaml_file, 'r') as in_fd:
            yaml_dict = yaml.load(in_fd)
        return cls(**yaml_dict)

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

    def write_to_file(self, out_file):
        out_dict = self.to_dict()
        with open(out_file, 'w') as out_fd:
            yaml.dump(out_dict, out_fd)

    def add_to_parser(self, parser):
        for k in self.next_key():
            attr_type = type(getattr(self, k))
            parser.add_argument('--'+k, dest=k, type=attr_type)
        return


