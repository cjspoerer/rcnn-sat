# rcnn-sat

Code for feedforward and recurrent neural network models used in
[XXX (placeholder for preprint)](www.google.com).

The code has been tested with TensorFlow 1.13 and Python 3.6.

## Using the code

Build the Keras model by importing and calling the model function, passing the preprocessed image as the first argument.

The code snippet below shows how this works and a full example of extracting activations from a pre-trained model is given [here](https://github.com/cjspoerer/rcnn-sat/blob/master/restore_and_extract_activations.ipynb).

```python
from rcnn_sat.bl_net import bl_net

input_layer = tf.keras.layers.Input((128, 128, 3))
model = bl_net(input_layer, classes=565, cumulative_readout=True)
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
