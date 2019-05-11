import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model, Model
from keras.backend.tensorflow_backend import set_session
import logging
import os
from pathlib import Path
import tensorflow as tf
from typing import Optional
from tqdm import tqdm

from dataset import Dataset, DatasetContainer, BatchInput, BatchOutput
from keras_callback_lr_scheduler import KerasCBLRScheduler
from keras_callback_tf_summary import KerasCBSummaryWriter
from model_config import ModelConfig


class KerasModelController(object):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model: Optional[Model] = None
        self.datasets: Optional[DatasetContainer] = None
        self.sess = None
        self.summary_writer = None
        self.lr_scheduler = KerasCBLRScheduler(config)
        tf.logging.set_verbosity(tf.logging.ERROR)

    def __enter__(self):
        if self.sess:
            return
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        set_session(self.sess)

        self.load()
        self.summary_writer = KerasCBSummaryWriter(self.config) \
            if self.config.learner_log_dir else None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # prefer garbage collection
        if self.summary_writer:
            self.summary_writer = None
        if self.sess:
            self.sess = None

    def load(self):
        from model import create_model
        """either create an new instance or load from previously saved"""
        model_root = Path(self.config.weight_root)
        saved_models = list(model_root.glob('*.hdf5'))
        if saved_models:
            # found previous saved models, use the latest
            saved_models.sort(key=os.path.getmtime, reverse=True)
            self.model = load_model(saved_models[0])
            logging.info(f'restoring model from {saved_models[0]}')
        else:
            assert self.config.allow_weight_init, \
                f'no saved model in {model_root} and we are not allowed to initiate'
            model_root.mkdir(parents=True, exist_ok=True)
            self.model = create_model(self.config)
            logging.info(f'initiating model and saving to {model_root}')
        self.model.summary()
        assert self.model

    def load_datasets(self, force_reload=False):
        if self.datasets and not force_reload:
            return
        self.datasets = DatasetContainer()
        self.datasets.train = Dataset.load_datasets(self.config, in_root=self.config.dataset_path)
        # TODO
        self.datasets.val = self.datasets.train

    def train(self, **kwargs):
        with self:
            config = self.config
            self.load_datasets()
            assert isinstance(self.datasets.train, Dataset)

            batch_size = min(config.batch_size, self.datasets.train.max_batch_size())
            assert batch_size > 0, f'train data size is {self.datasets.train.max_batch_size()}'
            config.iterations_per_epoch = len(self.datasets.train)//batch_size
            logging.info(self.config)

            train_generator = self.datasets.train.batch_generator(
                config.batch_size,
                augmentation=config.use_augmentation,
                loop=True,
                return_raw=True)
            val_generator = self.datasets.val.batch_generator(
                config.batch_size_inference,
                random_sampling=False,
                augmentation=False,
                loop=True,
                return_raw=True)
            early_stopping = EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=config.early_stop)
            model_cp = ModelCheckpoint(filepath=config.weight_root+'/checkpoint.hdf5',
                                       monitor='loss' if config.save_weight_on_best_train else 'val_loss',
                                       mode='min',
                                       save_best_only=True)
            callbacks = [self.lr_scheduler, early_stopping, model_cp]
            if self.summary_writer:
                callbacks.append(self.summary_writer)
            self.model.fit_generator(
                train_generator,
                steps_per_epoch=config.iterations_per_epoch,
                epochs=config.training_epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=val_generator,
                validation_steps=len(self.datasets.val)//config.batch_size_inference,
                )

    def evaluate(self, dataset):
        config = self.config
        batch_size = min(config.batch_size_inference, dataset.max_batch_size())
        num_class = self.config.board_size * self.config.board_size
        total_errors, total_samples = 0, 0
        # (predicted, target)
        confusion_matrix = np.zeros((num_class, num_class), dtype=int)
        with self:
            logging.info(f'evaluating on {dataset}')
            for batch_input in tqdm(dataset.batch_generator(batch_size,
                                                            random_sampling=False,
                                                            loop=False,
                                                            augmentation=False)):
                target_ys = batch_input.batch_ys.astype(int)
                batch_out = self.infer(batch_input)
                predicted_ys = np.argmax(batch_out.result, axis=1)
                errors = (target_ys != predicted_ys).astype(int)
                total_errors += np.sum(errors)
                total_samples += len(batch_input.batch_xs)
                np.add.at(confusion_matrix, (predicted_ys, target_ys), 1)
        errors = 1.0 - confusion_matrix.diagonal()/confusion_matrix.sum(axis=0)
        return {
            'classification_error': total_errors/total_samples,
            'confusion_matrix': confusion_matrix,
            'errors': errors,
            'score': float(1.0 - np.mean(errors)),
        }

    def infer(self, batch_input: BatchInput) -> BatchOutput:
        assert self.model, f'please use with statement to create session first'
        batch_out = BatchOutput()
        batch_out.result = self.model.predict_on_batch(batch_input.batch_xs)
        return batch_out




