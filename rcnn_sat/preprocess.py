import numpy as np
import tensorflow as tf

def preprocess_image(image):
    '''Scales pixel values to correct range before passing to the network

    Args:
        image: image to be passed to the network, should be an unscaled image in
            uint8 format (values in range [0, 255]).
    Returns:
        preprocessed_image
    '''

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError('image should be uint8')
    elif tf.contrib.framework.is_tensor(image):
        tf.assert_type(image, tf.uint8, message='image should be uint8')
    else:
        raise TypeError('image should be tf.Tensor or np.ndarray')

    preprocessed_image = image / 127.5 - 1.

    return preprocessed_image
