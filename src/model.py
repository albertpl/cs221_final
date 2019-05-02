from keras import layers
from keras import models
from keras import backend as K
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Nadam, Adam

from model_config import ModelConfig
from resnet import ResnetBuilder


def build_simple_model(config: ModelConfig):
    num_action = config.board_size * config.board_size
    # H x W x C=1 (fake dimension), W=max_seq_len, C=dim_x
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(config.board_size, config.board_size, config.feature_channel)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    # model.add(layers.Dropout(1.0 - config.dropout_keep_prob))
    model.add(layers.Dense(config.dense_layer_dim, activation='relu'))
    model.add(layers.Dense(config.dense_layer_dim, activation='relu'))
    model.add(layers.Dense(config.dense_layer_dim, activation='relu'))
    model.add(layers.Dense(num_action, activation='softmax'))
    return model


def build_resnet(config: ModelConfig):
    num_action = config.board_size * config.board_size
    resnet = ResnetBuilder.build_resnet_18(
        input_shape=(config.board_size, config.board_size, config.feature_channel),
        num_outputs=config.dense_layer_dim)
    output = layers.Dense(num_action, activation='softmax')(resnet.output)
    return models.Model(inputs=resnet.input, outputs=output)


def create_model(config: ModelConfig):
    # model = build_resnet(config)
    model = build_simple_model(config)
    model.compile(optimizer=Adam(lr=1e-4),
                  loss=sparse_categorical_crossentropy,
                  metrics=[sparse_categorical_accuracy, ])
    return model
