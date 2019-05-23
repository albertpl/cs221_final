from keras import layers
from keras import models
from keras import backend as K
from keras.optimizers import Nadam, Adam

from model_config import ModelConfig
from resnet import ResnetBuilder


def build_simple_model(config: ModelConfig):
    num_action = config.action_space_size
    # H x W x C=1 (fake dimension), W=max_seq_len, C=dim_x
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(config.board_size, config.board_size, config.feature_channel)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    # model.add(layers.Dropout(1.0 - config.dropout_keep_prob))
    model.add(layers.Dense(config.fc1_dim, activation='relu'))
    model.add(layers.Dense(config.fc2_dim, activation='relu'))
    x = model.output
    policy_network = layers.Dense(num_action, activation='softmax', name='policy')(x)
    value_network = layers.Dense(1, activation='tanh', name='value')(x)
    return models.Model(inputs=model.input, outputs=[policy_network, value_network])


def build_resnet(config: ModelConfig):
    num_action = config.action_space_size
    input_layer = layers.Input(shape=(config.board_size, config.board_size, config.feature_channel))
    resnet_out = ResnetBuilder.build_resnet_18(input_layer)
    x = layers.Dropout(0.2)(layers.Dense(config.fc1_dim, activation='relu')(resnet_out))
    x = layers.Dense(config.fc2_dim, activation='relu')(x)
    policy_network = layers.Dense(num_action, activation='softmax', name='policy')(x)
    value_network = layers.Dense(1, activation='tanh', name='value')(x)
    return models.Model(inputs=input_layer, outputs=[policy_network, value_network])


def create_model(config: ModelConfig):
    assert config.model_name in ('policy_with_baseline', ), config.model_name
    model = build_resnet(config)
    # model = build_simple_model(config)
    model.compile(optimizer=Adam(lr=1e-4),
                  metrics={'policy': 'categorical_accuracy'},
                  loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'})
    return model
