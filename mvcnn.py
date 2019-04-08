from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import resnet_model


import tensorflow as tf

slim = tf.contrib.slim


RESNET_SIZE = 18


class ModelnetModel(resnet_model.Model):
    def __init__(self, resnet_size, data_format='channels_last', num_classes=10,
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

        super(ModelnetModel, self).__init__(
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


def view_pooling(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    # print('vp before reducing:', vp.get_shape().as_list())
    vp = tf.reduce_max(vp, axis=[0], name=name)

    return vp


def mvcnn(inputs,
          num_classes,
          is_training=True,
          keep_prob=0.8,
          reuse=tf.AUTO_REUSE,
          scope='mvcnn'):
    '''
    :param inputs: N x V x H x W x C tensor
    :return:
    '''
    # resnet 18
    model = ModelnetModel(RESNET_SIZE, num_classes=num_classes)

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, 'mvcnn', [inputs], reuse=reuse):
        view_pool = []
        for i in range(n_views):
            view_batches = tf.gather(views, i)  # N x H x W x C

            net = model(view_batches, training=is_training)
            view_pool.append(net)

        # max pooling
        net = view_pooling(view_pool, 'view_pooling')
        # (?,7,7,512)
        net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_average_pooling')
        # (?,1,1,512)
        net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout')
        net = slim.flatten(net, scope='pre_logits_flatten')
        # (?,512)
        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')

    return logits, net



