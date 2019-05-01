from dataset import Dataset, BatchInput
import logging
from model_config import ModelConfig
from tqdm import tqdm


def test_dataset(in_root):
    config = ModelConfig()
    dataset = Dataset.load_datasets(config=config, in_root=in_root)
    assert dataset

    for batch in tqdm(dataset.batch_generator(batch_size=64)):
        assert isinstance(batch, BatchInput), type(batch)
        assert len(batch.batch_ys) == len(batch.batch_xs), batch
        assert batch.batch_xs[0].shape[0] == config.board_size, batch.batch_xs[0].shape
        assert batch.batch_xs[0].shape[1] == config.board_size, batch.batch_xs[0].shape
        assert batch.batch_xs[0].shape[2] == config.feature_channel, batch.batch_xs[0].shape


logging_format = '%(asctime)s %(module)s:%(lineno)d %(funcName)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
test_dataset('/tmp/go_games/')
