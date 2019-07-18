from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# from nets import resnet_model
from nets import resnet_v2

import tensorflow as tf

slim = tf.contrib.slim


# RESNET_SIZE = 34

#
# class ModelnetModel(resnet_model.Model):
#     def __init__(self, resnet_size, data_format='channels_last', num_classes=10,
#                  resnet_version=resnet_model.DEFAULT_VERSION,
#                  dtype=resnet_model.DEFAULT_DTYPE):
#         """These are the parameters that work for Imagenet data.
#
#             Args:
#               resnet_size: The number of convolutional layers needed in the model.
#               data_format: Either 'channels_first' or 'channels_last', specifying which
#                 data format to use when setting up the model.
#               num_classes: The number of output classes needed from the model. This
#                 enables users to extend the same model to their own datasets.
#               resnet_version: Integer representing which version of the ResNet network
#                 to use. See README for details. Valid values: [1, 2]
#               dtype: The TensorFlow dtype to use for calculations.
#             """
#
#         # For bigger models, we want to use "bottleneck" layers
#         if resnet_size < 50:
#             bottleneck = False
#         else:
#             bottleneck = True
#
#         super(ModelnetModel, self).__init__(
#             resnet_size=resnet_size,
#             bottleneck=bottleneck,
#             num_classes=num_classes,
#             num_filters=64,
#             kernel_size=7,
#             conv_stride=2,
#             first_pool_size=3,
#             first_pool_stride=2,
#             block_sizes=_get_block_sizes(resnet_size),
#             block_strides=[1, 2, 2, 2],
#             resnet_version=resnet_version,
#             data_format=data_format,
#             dtype=dtype
#         )
#
#
# def _get_block_sizes(resnet_size):
#     """Retrieve the size of each block_layer in the ResNet model.
#
#     The number of block layers used for the Resnet model varies according
#     to the size of the model. This helper grabs the layer set we want, throwing
#     an error if a non-standard size has been selected.
#
#     Args:
#       resnet_size: The number of convolutional layers needed in the model.
#
#     Returns:
#       A list of block sizes to use in building the model.
#
#     Raises:
#       KeyError: if invalid resnet_size is received.
#     """
#     choices = {
#         18: [2, 2, 2, 2],
#         34: [3, 4, 6, 3],
#         50: [3, 4, 6, 3],
#         101: [3, 4, 23, 3],
#         152: [3, 8, 36, 3],
#         200: [3, 24, 36, 3]
#     }
#
#     try:
#         return choices[resnet_size]
#     except KeyError:
#         err = ('Could not find layers for selected Resnet size.\n'
#                'Size received: {}; sizes allowed: {}.'.format(
#             resnet_size, choices.keys()))
#         raise ValueError(err)


# # TODO
# def encode(preprocess_fn, network_factory, checkpoint_path, images_or_filenames,
#            batch_size=32, session=None, image_shape=None):
#     """
#
#     Parameters
#     ----------
#     preprocess_fn : Callable[tf.Tensor] -> tf.Tensor
#         A callable that applies preprocessing to a given input image tensor of
#         dtype tf.uint8 and returns a floating point representation (tf.float32).
#     network_factory : Callable[tf.Tensor] -> (tf.Tensor, tf.Tensor)
#         A callable that takes as argument a preprocessed input image of dtype
#         tf.float32 and returns the feature representation as well as a logits
#         tensors. The logits may be set to None if not required by the loss.
#     checkpoint_path : str
#         Checkpoint file to load.
#     images_or_filenames : List[str] | np.ndarray
#         Either a list of filenames or an array of images.
#     batch_size : Optional[int]
#         Optional batch size; defaults to 32.
#     session : Optional[tf.Session]
#         Optional TensorFlow session. If None, a new session is created.
#     image_shape : Tuple[int, int, int] | NoneType
#         Image shape (height, width, channels) or None. If None, `train_x` must
#         be an array of images such that the shape can be queries from this
#         variable.
#
#     Returns
#     -------
#     np.ndarray
#
#     """
#     if image_shape is None:
#         assert type(images_or_filenames) == np.ndarray
#         image_shape = images_or_filenames.shape[1:]
#     elif type(images_or_filenames) == np.ndarray:
#         assert images_or_filenames.shape[1:] == image_shape
#     read_from_file = type(images_or_filenames) != np.ndarray
#
#     encoder_fn = _create_encoder(
#         preprocess_fn, network_factory, image_shape, batch_size, session,
#         checkpoint_path, read_from_file)
#     features = encoder_fn(images_or_filenames)
#     return features
#
#
# # TODO
# def _create_encoder(preprocess_fn, network_factory, image_shape, batch_size=32,
#                     session=None, checkpoint_path=None, read_from_file=False):
#     if read_from_file:
#         num_channels = image_shape[-1] if len(image_shape) == 3 else 1
#         input_var = tf.placeholder(tf.string, (None, ))
#         image_var = tf.map_fn(
#             lambda x: tf.image.decode_jpeg(
#                 tf.read_file(x), channels=num_channels),
#             input_var, back_prop=False, dtype=tf.uint8)
#         image_var = tf.image.resize_images(image_var, image_shape[:2])
#     else:
#         input_var = tf.placeholder(tf.uint8, (None, ) + image_shape)
#         image_var = input_var
#
#     preprocessed_image_var = tf.map_fn(
#         lambda x: preprocess_fn(x, is_training=False),
#         image_var, back_prop=False, dtype=tf.float32)
#
#     feature_var, _ = network_factory(preprocessed_image_var)
#     feature_dim = feature_var.get_shape().as_list()[-1]
#
#     if session is None:
#         session = tf.Session()
#     if checkpoint_path is not None:
#         tf.train.get_or_create_global_step()
#         init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
#             checkpoint_path, slim.get_model_variables())
#         session.run(init_assign_op, feed_dict=init_feed_dict)
#
#     def encoder(data_x):
#         out = np.zeros((len(data_x), feature_dim), np.float32)
#         queued_trainer.run_in_batches(
#             lambda x: session.run(feature_var, feed_dict=x),
#             {input_var: data_x}, out, batch_size)
#         return out
#
#     return encoder


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
          keep_prob=0.6,
          reuse=tf.compat.v1.AUTO_REUSE,
          attention_module=None,
          scope='mvcnn'):
    '''
    :param inputs: N x V x H x W x C tensor
    :return:
    '''
    # model = ModelnetModel(resnet_size=RESNET_SIZE, num_classes=num_classes)

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, 'mvcnn', [inputs], reuse=reuse):
        view_pool = []
        for i in range(n_views):
            view_batches = tf.gather(views, i)  # N x H x W x C

            # net = model(view_batches, training=is_training)
            net, _ = \
                resnet_v2.resnet_v2_50(inputs,
                                       num_classes=num_classes,
                                       is_training=is_training,
                                       attention_module=attention_module,
                                       scope='resnet_v2_50')

            view_pool.append(net)

        # max pooling
        net = view_pooling(view_pool, 'view_pooling')
        # (?,7,7,512)
        net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_average_pooling')
        # (?,1,1,512)
        net = slim.flatten(net, scope='pre_logits_flatten')
        net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout')
        # (?,512)
        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')

    return logits


def mvcnn_with_deep_cosine_metric_learning(inputs,
                                           num_classes,
                                           is_training=True,
                                           keep_prob=0.6,
                                           reuse=tf.compat.v1.AUTO_REUSE,
                                           attention_module=None,
                                           scope='mvcnn'):
    '''
    :param inputs: N x V x H x W x C tensor
    :return:
    '''
    # resnet 18
    # model = ModelnetModel(RESNET_SIZE, num_classes=num_classes)

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.compat.v1.variable_scope(scope, 'mvcnn', [inputs], reuse=reuse):
        fc_regularizer = slim.l2_regularizer(1e-8)

        view_pool = []
        for i in range(n_views):
            view_batches = tf.gather(views, i)  # N x H x W x C

            # net = model(view_batches, training=is_training)
            net, _ = \
                resnet_v2.resnet_v2_50(view_batches,
                                       num_classes=num_classes,
                                       is_training=is_training,
                                       attention_module=attention_module,
                                       scope='resnet_v2_50')
            view_pool.append(net)

        # max pooling
        net = view_pooling(view_pool, 'view_pooling')

        ###############################
        # deep cosine metric learning
        ###############################
        # (?,7,7,512)
        feature_dim = net.get_shape().as_list()[-1]
        net = slim.flatten(net)
        net = slim.dropout(net, keep_prob=keep_prob)
        net = slim.fully_connected(net,
                                   feature_dim,
                                   normalizer_fn=slim.batch_norm,
                                   weights_regularizer=fc_regularizer,
                                   scope='fc1')

        features = net

        # Features in rows, normalize axis 1.
        # The final l2 normalization projects features onto the unit hypersphere
        # for application of the cosine softmax classifier.
        features = tf.nn.l2_normalize(features, axis=1)

        with tf.compat.v1.variable_scope("ball", reuse=reuse):
            weights = \
                slim.model_variable("mean_vectors",
                                    (feature_dim, int(num_classes)),
                                    initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                    regularizer=None)
            # The scaling parameter κ controls
            # the shape of the conditional class probabilities
            scale = \
                slim.model_variable("scale",
                                    (),
                                    tf.float32,
                                    initializer=tf.constant_initializer(0., tf.float32),
                                    regularizer=slim.l2_regularizer(1e-1))

            tf.compat.v1.summary.scalar("scale", scale)
            scale = tf.nn.softplus(scale)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = tf.nn.l2_normalize(weights, axis=0)
        logits = scale * tf.matmul(features, weights_normed)

    return logits, features  # use it for retrieval.



