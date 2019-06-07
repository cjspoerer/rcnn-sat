'''
Keras implementation of BL network
'''

import tensorflow as tf


class BLConvLayer(object):
    '''BL recurrent convolutional layer

    Note that this is NOT A KERAS LAYER but is an object containing Keras layers

    Args:
        filters: Int, number of output filters in convolutions
        kernel_size: Int or tuple/list of 2 integers, specifying the height and
            width of the 2D convolution window. Can be a single integer to
            specify the same value for all spatial dimensions.
        layer_name: String, prefix for layers in the RCL
        '''
    def __init__(self, filters, kernel_size, layer_name):
        # initialise convolutional layers
        self.b_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            name='{}_BConv'.format(layer_name))

        self.l_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            name='{}_LConv'.format(layer_name))

        # layer for summing convolutions
        self.sum_convs = tf.keras.layers.Lambda(
            tf.add_n, name='{}_ConvSum'.format(layer_name))

        # holds the most recent bottom-up conv
        # useful when the bottom-up input does not change, e.g. input image
        self.previous_b_conv = None

    def __call__(self, b_input=None, l_input=None):
        conv_list = []

        if not b_input is None:
            # run bottom-up conv and save result
            conv_list.append(self.b_conv(b_input))
            self.previous_b_conv = conv_list[-1]
        elif not self.previous_b_conv is None:
            # use the most recent bottom-up conv
            conv_list.append(self.previous_b_conv)
        else:
            raise ValueError('b_input must be given on first pass')
        
        # run lateral conv
        if l_input is not None:
            conv_list.append(self.l_conv(l_input))

        # return element-wise sum of convolutions
        return self.sum_convs(conv_list)


def bl_net(input_tensor, classes, n_timesteps=8, cumulative_readout=False):
        '''Build the computational graph for the model

        Note that evaluations based on model outputs will reflect instantaneous
        rather than cumulative readouts

        Args:
            input_tensor: Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: int, number of classes to classify images into
            n_timesteps: int, number of model time steps to build
            cumulative_readout: Bool, if True then the outputs correspond to a
                cumulative readout on each time step if True then they
                correspond to a instant readout

        Returns:
            model
        '''

        data_format = tf.keras.backend.image_data_format()
        norm_axis = -1 if data_format == 'channels_last' else -3

        # initialise trainable layers (RCLs and linear readout)
        layers = [
            BLConvLayer(96, 7, 'RCL_0'),
            BLConvLayer(128, 5, 'RCL_1'),
            BLConvLayer(192, 3, 'RCL_2'),
            BLConvLayer(256, 3, 'RCL_3'),
            BLConvLayer(512, 3, 'RCL_4'),
            BLConvLayer(1024, 3, 'RCL_5'),
            BLConvLayer(2048, 1, 'RCL_6'),
        ]
        readout_dense = tf.keras.layers.Dense(
                classes, kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                name='ReadoutDense')

        # initialise list for activations and outputs
        n_layers = len(layers)
        activations = [[None for _ in range(n_layers)]
                       for _ in range(n_timesteps)]
        presoftmax = [None for _ in range(n_timesteps)]
        outputs = [None for _ in range(n_timesteps)]

        # build the model
        for t in range(n_timesteps):
            for n, layer in enumerate(layers):

                # get the bottom-up input
                if n == 0:
                    # B conv on the image does not need to be recomputed
                    b_input = input_tensor if t == 0 else None
                else:
                    # pool b_input for all layers apart from input
                    b_input = tf.keras.layers.MaxPool2D(
                        pool_size=(2, 2),
                        name='MaxPool_Layer_{}_Time_{}'.format(n, t)
                        )(activations[t][n-1])

                # get the lateral input
                if t == 0:
                    l_input = None
                else:
                    l_input = activations[t-1][n]

                # convolutions
                x_tn = layer(b_input, l_input)
                # batch-normalisation
                x_tn = tf.keras.layers.BatchNormalization(
                    norm_axis,
                    name='BatchNorm_Layer_{}_Time_{}'.format(n, t))(x_tn)
                # ReLU
                activations[t][n] = tf.keras.layers.Activation(
                    'relu', name='ReLU_Layer_{}_Time_{}'.format(n, t))(x_tn)

            # add the readout layers
            x = tf.keras.layers.GlobalAvgPool2D(
                name='GlobalAvgPool_Time_{}'.format(t)
                )(activations[t][-1])
            presoftmax[t] = readout_dense(x)
            
            # select cumulative or instant readout
            if cumulative_readout and t > 0:
                x = tf.keras.layers.Add(
                    name='CumulativeReadout_Time_{}'.format(t)
                    )(presoftmax[:t+1])
            else:
                x = presoftmax[t]
            outputs[t] = tf.keras.layers.Activation(
                'softmax', name='Sotfmax_Time_{}'.format(t))(x)

        # create Keras model and return
        model = tf.keras.Model(
            inputs=input_tensor,
            outputs=outputs,
            name='bl_net')    

        return model
