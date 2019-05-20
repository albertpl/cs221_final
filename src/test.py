import sys
from dataset import DatasetContainer, Dataset
import logging
from model_config import ModelConfig
import numpy as np
from tqdm import tqdm


def test_dataset(config: ModelConfig, dataset: Dataset):
    assert dataset is not None
    for batch_input, batch_output in tqdm(dataset.batch_generator(batch_size=config.batch_size)):
        assert isinstance(batch_input, np.ndarray), type(batch_input)
        assert isinstance(batch_output, dict), type(batch_output)
        for key_str in ('policy', 'value'):
            assert key_str in batch_output, f'missing {key_str} in {batch_output}'
        assert batch_input.ndim == 4, batch_input.shape
        assert batch_input.shape[0] == config.batch_size, batch_input.shape
        assert batch_input.shape[1] == config.board_size, batch_input.shape
        assert batch_input.shape[2] == config.board_size, batch_input.shape
        assert batch_input.shape[3] == config.feature_channel, batch_input.shape
        policy = batch_output['policy']
        assert isinstance(policy, np.ndarray), type(policy)
        assert policy.ndim == 2, policy.shape
        assert policy.shape[0] == config.batch_size, policy.shape
        assert policy.shape[1] == config.action_space_size, policy.shape
        assert np.all(policy <= 1.0), np.max(policy)
        assert np.all(-1.0 <= policy), np.min(policy)
        value = batch_output['value']
        assert isinstance(value, np.ndarray), type(value)
        assert value.ndim == 1, value.shape
        assert value.shape[0] == config.batch_size, value.shape
        assert np.all((value == 1) + (value == -1)), value


def test_datasets(in_root):
    config = ModelConfig()
    container = DatasetContainer.from_path(config=config, in_root=in_root)
    assert container, in_root
    test_dataset(config, container.train)
    for dataset_str in ('val', 'test'):
        dataset = getattr(container, dataset_str)
        if dataset is None:
            print(f'skipping {dataset_str}')
            continue
        test_dataset(config, dataset)


logging_format = '%(asctime)s %(module)s:%(lineno)d %(funcName)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
if len(sys.argv) != 2:
    print('missing dataset path')
    sys.exit()
test_datasets(sys.argv[1])
