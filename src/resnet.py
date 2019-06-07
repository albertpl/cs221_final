""" adapted from
https://github.com/raghakot/keras-resnet/blob/master/resnet.py
"""
from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    padding = conv_params.setdefault("padding", "same")
    # kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    # kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=False
                      )(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    padding = conv_params.setdefault("padding", "same")
    # kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    # kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      )(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          )(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            input = block_function(filters=filters,
                                   init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input
    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           )(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return add([input, residual])

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              )(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters,
                                     kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters,
                                 kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4,
                                 kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_layer, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_layer: The input layer in the form (nb_rows, nb_cols, nb_channels)
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        assert K.image_dim_ordering() == 'tf'

        # Load function from str if needed.
        block_fn = _get_block(block_fn)
        conv1 = _conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(1, 1))(input_layer)
        # pool1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")(conv1)

        block = conv1
        filters = 32
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            # filters *= 2

        # Last activation
        block = _bn_relu(block)
        return block

    @staticmethod
    def build_custom_resnet(input_layer, num_block):
        def res_layer(x):
            init_output = _conv_bn_relu(filters=32, kernel_size=3, strides=(1, 1))(x)
            conv = Conv2D(filters=32,
                          kernel_size=3,
                          padding='same',
                          use_bias=False
                          )(init_output)
            return Activation('relu')(add([x, BatchNormalization(axis=-1)(conv)]))
        shared_output = _conv_bn_relu(filters=32, kernel_size=3, strides=(1, 1))(input_layer)
        for _ in range(num_block):
            shared_output = res_layer(shared_output)
        return shared_output

    @staticmethod
    def build_resnet_18(input_layer):
        return ResnetBuilder.build(input_layer, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_layer):
        return ResnetBuilder.build(input_layer, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_layer):
        return ResnetBuilder.build(input_layer, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_layer):
        return ResnetBuilder.build(input_layer, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_layer):
        return ResnetBuilder.build(input_layer, bottleneck, [3, 8, 36, 3])