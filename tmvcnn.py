from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def tmvcnn(inputs, n_classes, is_training=True, keep_prob=0.8):
    '''
    :param inputs: N x V x H x W x C tensor
    :return:
    '''
    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])
    for i in range(n_views):
        view_batches = tf.gather(views, i)  # N x H x W x C
        # TODO resnet 18

    return