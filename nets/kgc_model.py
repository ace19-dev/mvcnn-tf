from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from resnet import resnet_model


DEFAULT_IMAGE_SIZE = 224
NUM_CLASSES = 3

class KGCModel(resnet_model.Model):
    def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE):
        """These are the parameters that work for Imagenet data.

            Args:
              resnet_size: The number of convolutional layers needed in the model.
              data_format: Either 'channels_first' or 'channels_last', specifying which
                data format to use when setting up the model.
              num_classes: The number of output classes needed from the model. This
                enables users to extend the same model to their own datasets.
              resnet_version: Integer representing which version of the ResNet network
                to use. See README for details. Valid values: [1, 2]
              dtype: The TensorFlow dtype to use for calculations.
            """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        super(KGCModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)