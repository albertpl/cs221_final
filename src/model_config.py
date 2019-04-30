def add_to_parser(config_obj, parser):
    for k in config_obj.next_key():
        attr_type = type(getattr(config_obj, k))
        parser.add_argument('--'+k, dest=k, type=attr_type)
    return


class ModelConfig(object):
    board_size = 19
    game_index_file = 'game_index.yaml'

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
