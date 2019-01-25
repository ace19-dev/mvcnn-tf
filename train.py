from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

from utils import train_utils
import tmvcnn
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

flags.DEFINE_float('learning_rate', .0001,
                   'The learning rate for model training.')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# Settings for fine-tuning the network.
flags.DEFINE_string('pre_trained_checkpoint',
                    # './pre-trained/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt',
                    None,
                    'The pre-trained checkpoint in tensorflow format.')
flags.DEFINE_string('checkpoint_exclude_scopes',
                    'resnet_v2_101/logits',
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
                    'resnet_v2_101',
                    'The name of the architecture to train.')
flags.DEFINE_boolean('ignore_missing_vars',
                     False,
                     'When restoring a checkpoint would ignore missing variables.')

flags.DEFINE_string('dataset_dir',
                    # '/home/ace19/dl_data/kgc',
                    '/home/ace19/dl_data/modelnet', # temporary source path
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 100,
                     'How many training loops to run')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels',
                    'airplane,bed,bookshelf,cone,person,toilet,vase',
                    'Labels to use')


KGC_TRAINING_DATA_SIZE = 2780


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    tf.logging.info('Creating train logdir: %s', FLAGS.train_logdir)

    with tf.Graph().as_default() as graph:
        global_step = tf.train.get_or_create_global_step()

        X = tf.placeholder(tf.float32,
                           [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3])
        ground_truth = tf.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.placeholder(tf.bool)
        dropout_keep_prob = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32)

        logits = tmvcnn.tmvcnn(X, num_classes, dropout_keep_prob)

        tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=logits)

        # Gather update ops. These contain, for example, the updates for the
        # batch_norm variables created by model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        predition = tf.argmax(logits, 1, name='prediction')
        correct_predition = tf.equal(predition, ground_truth)
        confusion_matrix = tf.confusion_matrix(ground_truth,
                                               predition,
                                               num_classes=num_classes)
        accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        summaries.add(tf.summary.scalar('accuracy', accuracy))

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan')
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')

        # Add the summaries. These contain the summaries created by model
        # and either optimize() or _gather_loss()
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph)

        #####################
        # prepare data
        #####################
        tfrecord_names = tf.placeholder(tf.string, shape=[])
        training_dataset = data.Dataset(tfrecord_names, FLAGS.height, FLAGS.width,
                                        batch_size=FLAGS.batch_size)
        iterator = training_dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
            if FLAGS.pre_trained_checkpoint:
                train_utils.restore_fn(FLAGS)

            start_epoch = 0
            batches = int(KGC_TRAINING_DATA_SIZE / FLAGS.batch_size)
            if KGC_TRAINING_DATA_SIZE % FLAGS.batch_size > 0:
                batches += 1

            # The filenames argument to the TFRecordDataset initializer can either
            # be a string, a list of strings, or a tf.Tensor of strings.
            tf_filenames = os.path.join(FLAGS.dataset_dir, 'train.record')
            ##################
            # Training loop.
            ##################
            for n_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                tf.logging.info('--------------------------')
                tf.logging.info(' Epoch %d' % n_epoch)
                tf.logging.info('--------------------------')

                sess.run(iterator.initializer, feed_dict={tfrecord_names: tf_filenames})
                for step in range(batches):
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

                    train_summary, train_accuracy, train_loss, _ = \
                        sess.run([summary_op, accuracy, total_loss, train_op],
                        feed_dict={X: train_batch_xs,
                                   ground_truth: train_batch_ys,
                                   learning_rate: FLAGS.learning_rate,
                                   is_training: True,
                                   dropout_keep_prob: 0.8})

                    train_writer.add_summary(train_summary, n_epoch)
                    tf.logging.info('Epoch #%d, Step #%d, rate %.10f, accuracy %.1f%%, loss %f' %
                                    (n_epoch, step, FLAGS.learning_rate, train_accuracy * 100, train_loss))

                ###################################################
                # TODO: Validate the model on the validation set
                ###################################################

                # Save the model checkpoint periodically.
                if (n_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.logging.info('Saving to "%s-%d"', checkpoint_path, n_epoch)
                    saver.save(sess, checkpoint_path, global_step=n_epoch)


if __name__ == '__main__':
    tf.app.run()