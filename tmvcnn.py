from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import kgc_model


import tensorflow as tf


RESNET_SIZE = 18


def _view_pooling(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    print('vp before reducing:', vp.get_shape().as_list())
    vp = tf.reduce_max(vp, [0], name=name)
    return vp


def tmvcnn(inputs,
           num_classes,
           dropout_keep_prob,
           is_training=True,
           reuse=tf.AUTO_REUSE,
           scope='t-mvcnn'):
    '''
    :param inputs: N x V x H x W x C tensor
    :return:
    '''
    # resnet 18
    model = kgc_model.KGCModel(RESNET_SIZE,
                               num_classes=num_classes)

    n_views = inputs.get_shape().as_list()[1]
    # transpose views: (NxVxHxWxC) -> (VxNxHxWxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    with tf.variable_scope(scope, 't-mvcnn', [inputs], reuse=reuse):
        view_pool = []
        for i in range(n_views):
            view_batches = tf.gather(views, i)  # N x H x W x C

            net = model(view_batches, training=is_training)
            view_pool.append(net)

        # TODO pooling, classification, ...
        logits = _view_pooling(view_pool, 'view_pooling')

    return logits



