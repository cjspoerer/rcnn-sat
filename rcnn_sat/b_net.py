'''
Keras implementation of B networks
'''

import tensorflow as tf


def b_layer(x, filters, kernel, layer_num, pooling=True):
    '''Base layer for B models
    '''
    if pooling:
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            name='MaxPool_Layer_{}'.format(layer_num))(x)

    x = tf.keras.layers.Conv2D(
        filters, kernel, padding='same', use_bias=False,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        name='Conv_Layer_{}'.format(layer_num))(x)

    data_format = tf.keras.backend.image_data_format()
    norm_axis = -1 if data_format == 'channels_last' else -3
    x = tf.keras.layers.BatchNormalization(
        norm_axis,
        name='BatchNorm_Layer_{}'.format(layer_num))(x)

    x = tf.keras.layers.Activation(
        'relu', name='ReLU_Layer_{}'.format(layer_num))(x)

    return x


def readout(x, classes):
    '''Readout layer
    '''
    x = tf.keras.layers.GlobalAvgPool2D(name='GlobalAvgPool')(x)
    x = tf.keras.layers.Dense(
        classes, kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        name='ReadoutDense')(x)
    x = tf.keras.layers.Activation('softmax', name='Softmax')(x)
    return x


def b_net(input_tensor, classes):
    '''Defines a B model
    '''
    x = b_layer(input_tensor, 96, 7, 0, pooling=False)
    x = b_layer(x, 128, 5, 1)
    x = b_layer(x, 192, 3, 2)
    x = b_layer(x, 256, 3, 3)
    x = b_layer(x, 512, 3, 4)
    x = b_layer(x, 1024, 3, 5)
    x = b_layer(x, 2048, 1, 6)
    output_tensor = readout(x, classes)

    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


def b_k_net(input_tensor, classes):
    '''Defines a B-K model
    '''
    x = b_layer(input_tensor, 96, 11, 0, pooling=False)
    x = b_layer(x, 128, 7, 1)
    x = b_layer(x, 192, 5, 2)
    x = b_layer(x, 256, 5, 3)
    x = b_layer(x, 512, 5, 4)
    x = b_layer(x, 1024, 5, 5)
    x = b_layer(x, 2048, 3, 6)
    output_tensor = readout(x, classes)

    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


def b_f_net(input_tensor, classes):
    '''Defines a B-F model
    '''
    x = b_layer(input_tensor, 192, 7, 0, pooling=False)
    x = b_layer(x, 256, 5, 1)
    x = b_layer(x, 384, 3, 2)
    x = b_layer(x, 512, 3, 3)
    x = b_layer(x, 1024, 3, 4)
    x = b_layer(x, 2048, 3, 5)
    x = b_layer(x, 4096, 1, 6)
    output_tensor = readout(x, classes)

    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)


def b_d_net(input_tensor, classes):
    '''Defines a B-D model
    '''

    x = b_layer(input_tensor, 96, 7, 0, pooling=False)
    x = b_layer(x, 96, 7, 1, pooling=False)
    x = b_layer(x, 128, 5, 2)
    x = b_layer(x, 128, 5, 3, pooling=False)
    x = b_layer(x, 192, 3, 4)
    x = b_layer(x, 192, 3, 5, pooling=False)
    x = b_layer(x, 256, 3, 6)
    x = b_layer(x, 256, 3, 7, pooling=False)
    x = b_layer(x, 512, 3, 8)
    x = b_layer(x, 512, 3, 9, pooling=False)
    x = b_layer(x, 1024, 3, 10)
    x = b_layer(x, 1024, 3, 11, pooling=False)
    x = b_layer(x, 2048, 1, 12)
    x = b_layer(x, 2048, 1, 13, pooling=False)
    output_tensor = readout(x, classes)

    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
