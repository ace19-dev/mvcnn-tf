'''
input: image - [1, FLAGS.num_views, FLAGS.height, FLAGS.width, 3]
output: nearest neighbor image. apply deep cosine metric
        TODO: how build a candiate images to compare in all data? or
        TODO: just find a one of them which is included input image categories?

When training is finished,
the classifier is stripped of the network and distance queries are
made using cosine similarity or Euclidean distance on the final layer of the network.

Deep metric learning approaches encode notion of similarity directly into the training objective.

When the feature encoder is trained with the classifier jointly by minimization of the
cross-entropy loss, the parameters of the encoder network are adapted to
push samples away from the decision boundary as far as possible.

Cosine Softmax Classifier

The final l2 normalization projects features onto the unit hypersphere
for application of the cosine softmax classifier.

'''

import datetime
import os
import csv

import tensorflow as tf

from retrieval import retrieval_data
import mvcnn

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


# Dataset settings.
flags.DEFINE_string('retrieval_src',
                    '/home/ace19/dl_data/modelnet/validate.record',
                    'Where the dataset reside.')

# flags.DEFINE_string('retrieval_target_root',
#                     '/home/ace19/dl_data/modelnet/retrieval',
#                     'The path where retrieval image is')

flags.DEFINE_string('checkpoint_path',
                    '../models',
                    'Directory where to read training checkpoints.')

flags.DEFINE_integer('batch_size', 10, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels',
                    'airplane,bed,bookshelf,toilet,vase',
                    'number of classes')


MODELNET_RETRIEVAL_SRC_DATA_SIZE = 350


# def get_targets():
#     views_dirs = os.listdir(FLAGS.retrieval_target_root)
#
#     for dir in views_dirs:
#         views_path = os.path.join(FLAGS.retrieval_target_root, dir)
#         views = os.listdir(views_path)


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    # Define the model
    X = tf.placeholder(tf.float32,
                       [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                       name='X')
    _, features = \
        mvcnn.mvcnn_with_deep_cosine_metric_learning(X,
                                                      num_classes,
                                                      is_training=False,
                                                      keep_prob=1.0)

    # Prepare retrieval source data
    filenames = tf.placeholder(tf.string, shape=[])
    _dataset = retrieval_data.Dataset(filenames,
                                      FLAGS.num_views,
                                      FLAGS.height,
                                      FLAGS.width,
                                      FLAGS.batch_size)
    iterator = _dataset.dataset.make_initializable_iterator()
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
        batches = int(MODELNET_RETRIEVAL_SRC_DATA_SIZE / FLAGS.batch_size)
        if MODELNET_RETRIEVAL_SRC_DATA_SIZE % FLAGS.batch_size > 0:
            batches += 1

        # Save or Load trained features
        retrieval_source_filenames = os.path.join(FLAGS.retrieval_src)
        sess.run(iterator.initializer, feed_dict={filenames: retrieval_source_filenames})

        features_repo = []
        for i in range(batches):
            batch_xs, filenames = sess.run(next_batch)
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

            # (10,512)
            _features = sess.run([features], feed_dict={X: batch_xs})
            features_repo.extend(_features)

        tf.logging.info("size %d" % len(features_repo))

        # get retrieval targets


if __name__ == '__main__':
    tf.app.run()