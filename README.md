# rcnn-sat

Code for feedforward and recurrent neural network models used in
the paper [Recurrent neural networks can explain flexible trading of speed and accuracy in biological vision](https://doi.org/10.1371/journal.pcbi.1008215).

The code has been tested with TensorFlow 1.13 and Python 3.6.

## Using the code

The following code snippet shows how to build the Keras model and generate a prediction for a random image. A full example of extracting activations from a pre-trained model is given [here](https://github.com/cjspoerer/rcnn-sat/blob/master/restore_and_extract_activations.ipynb).

```python
import numpy as np
import tensorflow as tf
tf.enable_v2_behavior()
from rcnn_sat import preprocess_image, bl_net

# create the model with randomly initialised weights
input_layer = tf.keras.layers.Input((128, 128, 3))
model = bl_net(input_layer, classes=565, cumulative_readout=True)

# predict a random image
img = np.random.randint(0, 256, [1, 128, 128, 3], dtype=np.uint8)
model.predict(preprocess_image(img)) # softmax for each time step
```

## Pre-trained model weights

The checkpoint files for pre-trained eco-set and imagenet models are hosted
[here](https://osf.io/mz9hw/).


**Notes on pre-trained models:**
- Pre-trained models expect 128 x 128 images as input.
- ImageNet models have `classes=1000` and ecoset models have `classes=565`
- BL models were trained with `cumulative_readout=False` but can be tested using either option
- Model predictions correspond to the order of the categories within the files in
[`pretrained_output_categories`](https://github.com/cjspoerer/rcnn-sat/pretrained_output_categories).
