import tensorflow as tf

def preprocess_image(image):
    '''Scales pixel values to correct range before passing to the network

    Args:
        image: image to be passed to the network, should be an unscaled image in
            uint8 format (values in range [0, 255]).
    Returns:
        preprocessed_image
    '''
    tf.assert_type(image, tf.uint8)
    float_image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
    preprocessed_image = (float_image - .5) * 2
    return preprocessed_image
