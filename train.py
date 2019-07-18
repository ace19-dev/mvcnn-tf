from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

from utils import train_utils
import model
import data


import tensorflow as tf

slim = tf.contrib.slim


flags = tf.app.flags

FLAGS = flags.FLAGS



flags.DEFINE_string('train_logdir', './models',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_string('ckpt_name_to_save', 'resnet_v2.ckpt',
                    'Name to save checkpoint file')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')
flags.DEFINE_string('summaries_dir', './models/train_logs',
                     'Where to save summary logs for TensorBoard.')

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('base_learning_rate', .001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 1e-3,
                   'The rate to decay the base learning rate.')
flags.DEFINE_float('learning_rate_decay_step', .2000,
                   'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_float('training_number_of_steps', 300000,
                   'The number of steps used for training.')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')
flags.DEFINE_integer('slow_start_step', 510,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', .00005,
                   'Learning rate employed during slow start.')

# Settings for fine-tuning the network.
flags.DEFINE_string('pre_trained_checkpoint',
                    './pre-trained/resnet_v2_50.ckpt',
                    # None,
                    'The pre-trained checkpoint in tensorflow format.')
flags.DEFINE_string('checkpoint_exclude_scopes',
                    'resnet_v2_50/logits,resnet_v2_50/SpatialSqueeze,resnet_v2_50/predictions',
                    # None,
                    'Comma-separated list of scopes of variables to exclude '
                    'when restoring from a checkpoint.')
flags.DEFINE_string('trainable_scopes',
                    # 'ssd_300_vgg/block4_box, ssd_300_vgg/block7_box, \
                    #  ssd_300_vgg/block8_box, ssd_300_vgg/block9_box, \
                    #  ssd_300_vgg/block10_box, ssd_300_vgg/block11_box',
                    None,
                    'Comma-separated list of scopes to filter the set of variables '
                    'to train. By default, None would train all the variables.')
flags.DEFINE_string('checkpoint_model_scope',
                    None,
                    'Model scope in the checkpoint. None if the same as the trained model.')
flags.DEFINE_string('model_name',
                    'resnet_v2_50',
                    'The name of the architecture to train.')
flags.DEFINE_boolean('ignore_missing_vars',
                     False,
                     'When restoring a checkpoint would ignore missing variables.')

flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/modelnet10',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 60,
                     'How many training loops to run')
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('num_views', 12, 'number of views')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels',
                    'bathtub,bed,chair,desk,dresser,monitor,night_stand,sofa,table,toilet',
                    'Labels to use')

# temporary constant
MODELNET_TRAIN_DATA_SIZE = 5293
MODELNET_VALIDATE_DATA_SIZE = 1000


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    tf.io.gfile.makedirs(FLAGS.train_logdir)
    tf.compat.v1.logging.info('Creating train logdir: %s', FLAGS.train_logdir)

    with tf.Graph().as_default() as graph:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        X = tf.compat.v1.placeholder(tf.float32,
                           [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                           name='X')
        ground_truth = tf.compat.v1.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_keep_prob')
        # learning_rate = tf.placeholder(tf.float32, name='lr')

        # metric learning
        logits, features = \
            model.mvcnn_with_deep_cosine_metric_learning(X,
                                                         num_classes,
                                                         is_training=is_training,
                                                         keep_prob=dropout_keep_prob,
                                                         attention_module='se_block')
        # logits, features = mvcnn.mvcnn(X, num_classes)

        cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=logits)
        tf.compat.v1.summary.scalar("cross_entropy_loss", cross_entropy)

        # Gather update ops. These contain, for example, the updates for the
        # batch_norm variables created by model.
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # Gather initial summaries.
        summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

        predition = tf.argmax(logits, 1, name='prediction')
        correct_predition = tf.equal(predition, ground_truth)
        confusion_matrix = tf.math.confusion_matrix(ground_truth,
                                               predition,
                                               num_classes=num_classes)
        # accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        # summaries.add(tf.summary.scalar('accuracy', accuracy))
        accuracy = slim.metrics.accuracy(tf.cast(predition, tf.int64),
                                             ground_truth)
        tf.compat.v1.summary.scalar("accuracy", accuracy)

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.compat.v1.summary.histogram(model_var.op.name, model_var))

        # Add summaries for losses.
        for loss in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOSSES):
            summaries.add(tf.compat.v1.summary.scalar('losses/%s' % loss.op.name, loss))

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        summaries.add(tf.compat.v1.summary.scalar('learning_rate', learning_rate))

        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        total_loss = tf.compat.v1.check_numerics(total_loss, 'Loss is inf or nan')
        summaries.add(tf.compat.v1.summary.scalar('total_loss', total_loss))

        # TensorBoard: How to plot histogram for gradients
        # grad_summ_op = tf.compat.v1.summary.merge([tf.compat.v1.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grads_and_vars])

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')

        # Add the summaries. These contain the summaries created by model
        # and either optimize() or _gather_loss()
        summaries |= set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.compat.v1.summary.merge(list(summaries))
        train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir, graph)
        validation_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/validation', graph)

        #####################
        # prepare data
        #####################
        tfrecord_names = tf.compat.v1.placeholder(tf.string, shape=[])
        _dataset = data.Dataset(tfrecord_names,
                                FLAGS.num_views,
                                FLAGS.height,
                                FLAGS.width,
                                FLAGS.batch_size)
        iterator = _dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        sess_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        with tf.compat.v1.Session(config=sess_config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=1.0)
            if FLAGS.pre_trained_checkpoint:
                train_utils.restore_fn(FLAGS)

            start_epoch = 0
            training_batches = int(MODELNET_TRAIN_DATA_SIZE / FLAGS.batch_size)
            if MODELNET_TRAIN_DATA_SIZE % FLAGS.batch_size > 0:
                training_batches += 1
            val_batches = int(MODELNET_VALIDATE_DATA_SIZE / FLAGS.batch_size)
            if MODELNET_VALIDATE_DATA_SIZE % FLAGS.batch_size > 0:
                val_batches += 1

            # The filenames argument to the TFRecordDataset initializer can either
            # be a string, a list of strings, or a tf.Tensor of strings.
            training_tf_filenames = os.path.join(FLAGS.dataset_dir, 'train.record')
            val_tf_filenames = os.path.join(FLAGS.dataset_dir, 'validate.record')
            ##################
            # Training loop.
            ##################
            for n_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                tf.compat.v1.logging.info('--------------------------')
                tf.compat.v1.logging.info(' Epoch %d' % n_epoch)
                tf.compat.v1.logging.info('--------------------------')

                sess.run(iterator.initializer, feed_dict={tfrecord_names: training_tf_filenames})
                for step in range(training_batches):
                    train_batch_xs, train_batch_ys = sess.run(next_batch)
                    # # Verify image
                    # assert not np.any(np.isnan(train_batch_xs))
                    # n_batch = train_batch_xs.shape[0]
                    # n_view = train_batch_xs.shape[1]
                    # for i in range(n_batch):
                    #     for j in range(n_view):
                    #         img = train_batch_xs[i][j]
                    #         # scipy.misc.toimage(img).show()
                    #         # Or
                    #         img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    #         cv2.imwrite('/home/ace19/Pictures/' + str(i) +
                    #                     '_' + str(j) + '.png', img)
                    #         # cv2.imshow(str(train_batch_ys[idx]), img)
                    #         cv2.waitKey(100)
                    #         cv2.destroyAllWindows()

                    lr, train_summary, train_accuracy, train_loss, _ = \
                        sess.run([learning_rate, summary_op, accuracy, total_loss, train_op],
                                 feed_dict={X: train_batch_xs,
                                            ground_truth: train_batch_ys,
                                            is_training: True,
                                            dropout_keep_prob: 0.8})

                    # lr, train_summary, train_accuracy, train_loss, grad_vals, _ = \
                    #     sess.run([learning_rate, summary_op, accuracy, total_loss, grad_summ_op, train_op],
                    #     feed_dict={X: train_batch_xs,
                    #                ground_truth: train_batch_ys,
                    #                is_training: True,
                    #                dropout_keep_prob: 0.8})

                    train_writer.add_summary(train_summary, n_epoch)
                    # train_writer.add_summary(grad_vals, n_epoch)
                    tf.compat.v1.logging.info('Epoch #%d, Step #%d, rate %.10f, accuracy %.1f%%, loss %f' %
                                    (n_epoch, step, lr, train_accuracy * 100, train_loss))

                ###################################################
                # Validate the model on the validation set
                ###################################################
                tf.compat.v1.logging.info('--------------------------')
                tf.compat.v1.logging.info(' Start validation ')
                tf.compat.v1.logging.info('--------------------------')

                # Reinitialize iterator with the validation dataset
                sess.run(iterator.initializer, feed_dict={tfrecord_names: val_tf_filenames})

                total_val_accuracy = 0
                validation_count = 0
                total_conf_matrix = None
                for step in range(val_batches):
                    validation_batch_xs, validation_batch_ys = sess.run(next_batch)

                    val_summary, val_accuracy, conf_matrix = \
                        sess.run([summary_op, accuracy, confusion_matrix],
                                 feed_dict={X: validation_batch_xs,
                                            ground_truth: validation_batch_ys,
                                            is_training: False,
                                            dropout_keep_prob: 1.0})

                    validation_writer.add_summary(val_summary, n_epoch)

                    total_val_accuracy += val_accuracy
                    validation_count += 1
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix


                total_val_accuracy /= validation_count
                tf.compat.v1.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
                tf.compat.v1.logging.info('Validation accuracy = %.1f%% (N=%d)' %
                                (total_val_accuracy * 100, MODELNET_VALIDATE_DATA_SIZE))

                # Save the model checkpoint periodically.
                if (n_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.compat.v1.logging.info('Saving to "%s-%d"', checkpoint_path, n_epoch)
                    saver.save(sess, checkpoint_path, global_step=n_epoch)


if __name__ == '__main__':
    tf.compat.v1.app.run()