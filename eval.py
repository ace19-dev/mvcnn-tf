# Given C classes in the classification task,
# the output of the last layer in our network architecture is a
# vector with C elements, i.e., V = {v 1 , v 2 , · · · , v C }. Each
# element represents the probability that the subject belongs
# to that category. And the category with the largest value is
# the category it belongs to


import datetime
import os
import csv

import tensorflow as tf

import data
import tmvcnn

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


NUM_GROUP = 8

# temporary constant
MODELNET_EVAL_DATA_SIZE = 540


# Dataset settings.
flags.DEFINE_string('dataset_path', '/home/ace19/dl_data/modelnet/test.record',
                    'Where the dataset reside.')

flags.DEFINE_string('checkpoint_path',
                    os.getcwd() + '/models',
                    'Directory where to read training checkpoints.')

flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels',
                    'airplane,bed,bookshelf,cone,person,toilet,vase',
                    'number of classes')

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    # Define the model
    X = tf.placeholder(tf.float32,
                       [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                       name='inputs')
    ground_truth = tf.placeholder(tf.int64, [None], name='ground_truth')

    logits = tmvcnn.tmvcnn(X, num_classes, is_training=False)

    # prediction = tf.nn.softmax(logits)
    # predicted_labels = tf.argmax(prediction, 1)

    prediction = tf.argmax(logits, 1, name='prediction')
    correct_prediction = tf.equal(prediction, ground_truth)
    confusion_matrix = tf.confusion_matrix(
        ground_truth, prediction, num_classes=num_classes)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ################
    # Prepare data
    ################
    filenames = tf.placeholder(tf.string, shape=[])
    eval_dataset = data.Dataset(filenames, FLAGS.height, FLAGS.width,
                                     batch_size=FLAGS.batch_size)
    iterator = eval_dataset.dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        # Create a saver object which will save all the variables
        saver = tf.train.Saver()
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        saver.restore(sess, checkpoint_path)

        global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        # Get the number of training/validation steps per epoch
        batches = int(MODELNET_EVAL_DATA_SIZE / FLAGS.batch_size)
        if MODELNET_EVAL_DATA_SIZE % FLAGS.batch_size > 0:
            batches += 1

        ##################################################
        # prediction & make results into csv file.
        ##################################################
        start_time = datetime.datetime.now()
        print("Start prediction: {}".format(start_time))

        # id2name = {i: name for i, name in enumerate(labels)}
        # submission = {}

        eval_filenames = os.path.join(FLAGS.dataset_path)
        sess.run(iterator.initializer, feed_dict={filenames: eval_filenames})

        count = 0;
        total_acc = 0
        total_conf_matrix = None
        for i in range(batches):
            test_batch_xs, test_batch_ys = sess.run(next_batch)
            # # Verify image
            # n_batch = batch_xs.shape[0]
            # for i in range(n_batch):
            #     img = batch_xs[i]
            #     # scipy.misc.toimage(img).show()
            #     # Or
            #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
            #     # cv2.imshow(str(fnames), img)
            #     cv2.waitKey(100)
            #     cv2.destroyAllWindows()

            acc, conf_matrix = sess.run([accuracy, confusion_matrix],
                                             feed_dict={X: test_batch_xs,
                                                        ground_truth: test_batch_ys})

            total_acc += acc
            count += 1

            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        total_acc /= count
        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_acc * 100,
                                                                 MODELNET_EVAL_DATA_SIZE))

        end_time = datetime.datetime.now()
        tf.logging.info('#%d Data, End prediction: %s' % (MODELNET_EVAL_DATA_SIZE, end_time))
        tf.logging.info('prediction waste time: %s' % (end_time - start_time))


if __name__ == '__main__':
    tf.app.run()
