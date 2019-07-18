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

import cv2
import matplotlib.pyplot as plt

import datetime
import os
import csv
import numpy as np

import tensorflow as tf

from retrieval import retrieval_data, matching
import mvcnn


slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


# Dataset settings.
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/modelnet',
                    'Where the dataset reside.')

flags.DEFINE_string('output_dir',
                    '/home/ace19/dl_data/modelnet/retrieval_result',
                    'Where the dataset reside.')

flags.DEFINE_string('checkpoint_path',
                    '../models',
                    'Directory where to read training checkpoints.')

flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('num_views', 8, 'number of views')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels',
                    'airplane,bed,bookshelf,toilet,vase',
                    'number of classes')

# retrieval params
flags.DEFINE_float('max_cosine_distance', 0.2,
                   'Gating threshold for cosine distance')
flags.DEFINE_string('nn_budget', None,
                    'Maximum size of the appearance descriptors gallery. '
                    'If None, no budget is enforced.')


MODELNET_GALLERY_SIZE = 2525
MODELNET_QUERY_SIZE = 25

NUM_TOP = 3


# TODO
def display_retrieval(top_indices, gallery_path_list, query_path_list):
    tf.logging.info('write retrieval -> \n')
    top_indi_list = top_indices.tolist()
    for idx, indice in enumerate(top_indi_list):
        query = query_path_list[idx]
        tf.logging.info('query %d ==> %s ' % (idx, query))
        gallery = gallery_path_list[indice]
        tf.logging.info('gallery %d indice ==> %s ' % (indice, gallery))
        tf.logging.info('++++++++++++\n')

    # settings
    w, h = 30, 30  # for raster image
    columns = 8
    rows = 2

    tf.logging.info('display images -> \n')
    top_indi_list = top_indices.tolist()
    for idx, indice in enumerate(top_indi_list):
        fig = plt.figure(figsize=(w,h))
        # fig.suptitle("Query images and Gallery images", fontsize=16)

        query = query_path_list[idx]
        # subplot1 = fig.add_subplot(111)
        p = query[0].decode("utf-8")
        title = p.split('/')[-1].split('.')[0]
        # subplot1.set_title('query_' + str(idx) + 'th [' + title + ']')
        ax = []
        for i, path in enumerate(query):
            path = path.decode("utf-8")
            img = cv2.imread(path)
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i+1))
            label = path.split('/')[-1]
            ax[-1].set_title(label)  # set title
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        gallery = gallery_path_list[indice]
        # subplot2 = fig.add_subplot(111)
        # p2 = gallery[0].decode("utf-8")
        # title2 = p2.split('/')[-1].split('.')[0]
        # subplot2.set_title('gallery_' + str(indice) + 'th [' + title2 + ']')
        ax2 = []
        for j, path2 in enumerate(gallery):
            path2 = path2.decode("utf-8")
            img2 = cv2.imread(path2)
            # create subplot and append to ax
            ax2.append(fig.add_subplot(rows, columns, i+1+j+1))
            label2 = path2.split('/')[-1]
            ax2[-1].set_title(label2)  # set title
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        plt.tight_layout()
        plt.savefig(os.path.join(FLAGS.output_dir, 'query-' + title))
        plt.show()
        plt.close(fig)


# TODO: get value from indices
# TODO: get top-N indices
def match(galleries, queries):
    # The distance metric used for measurement to query.
    metric = matching.NearestNeighborDistanceMetric("cosine", FLAGS.max_cosine_distance)
    distance_matrix = metric.distance(queries, galleries)
    top_indice = np.argmin(distance_matrix, axis=1)

    # get value from indice
    # idx = np.argpartition(a, range(M))[:, :-M - 1:-1]  # topM_ind
    # out = a[np.arange(a.shape[0])[:, None], idx]  # topM_score
    # out_top1 = matrix[np.arange(matrix.shape[0])[:, None], top_indice]
    return top_indice

    # Top-N indices
    # top_indices = np.argpartition(matrix, NUM_TOP, axis=1)[:, :NUM_TOP]
    # return top_indices


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    # Define the model
    X = tf.placeholder(tf.float32,
                       [None, FLAGS.num_views, FLAGS.height, FLAGS.width, 3],
                       name='X')

    _, features = mvcnn.mvcnn_with_deep_cosine_metric_learning(X,
                                                               num_classes,
                                                               is_training=False,
                                                               keep_prob=1.0)

    # Prepare query source data
    tfrecord_filenames = tf.placeholder(tf.string, shape=[])
    _dataset = retrieval_data.Dataset(tfrecord_filenames,
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

        # global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        # Get the number of training/validation steps per epoch
        batches_gallery = int(MODELNET_GALLERY_SIZE / FLAGS.batch_size)
        if MODELNET_GALLERY_SIZE % FLAGS.batch_size > 0:
            batches_gallery += 1
        batches_query = int(MODELNET_QUERY_SIZE / FLAGS.batch_size)
        if MODELNET_QUERY_SIZE % FLAGS.batch_size > 0:
            batches_query += 1

        gallery_tf_filenames = os.path.join(FLAGS.dataset_dir, 'gallery.record')
        query_tf_filenames = os.path.join(FLAGS.dataset_dir, 'query.record')

        # TODO: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # TODO: It is better to create encode func which replace below codes
        # TODO: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # gallery images
        gallery_features_list = []
        gallery_path_list = []
        sess.run(iterator.initializer, feed_dict={tfrecord_filenames: gallery_tf_filenames})
        for i in range(batches_gallery):
            gallery_batch_xs, gallery_paths = sess.run(next_batch)
            # # Verify image
            # n_batch = gallery_batch_xs.shape[0]
            # for i in range(n_batch):
            #     img = gallery_batch_xs[i]
            #     # scipy.misc.toimage(img).show()
            #     # Or
            #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
            #     # cv2.imshow(str(fnames), img)
            #     cv2.waitKey(100)
            #     cv2.destroyAllWindows()

            # (10,512)
            _f = sess.run(features, feed_dict={X: gallery_batch_xs})
            gallery_features_list.extend(_f)
            gallery_path_list.extend(gallery_paths)

        # query images
        query_features_list = []
        query_path_list = []
        sess.run(iterator.initializer, feed_dict={tfrecord_filenames: query_tf_filenames})
        for i in range(batches_query):
            query_batch_xs, query_paths = sess.run(next_batch)
            # # Verify image
            # n_batch = query_batch_xs.shape[0]
            # for i in range(n_batch):
            #     img = query_batch_xs[i]
            #     # scipy.misc.toimage(img).show()
            #     # Or
            #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
            #     # cv2.imshow(str(fnames), img)
            #     cv2.waitKey(100)
            #     cv2.destroyAllWindows()

            # (10,512)
            _f = sess.run(features, feed_dict={X: query_batch_xs})
            query_features_list.extend(_f)
            query_path_list.extend(query_paths)

        # matching
        top_indices = match(gallery_features_list, query_features_list)

        # display top 5 image correspond to target
        display_retrieval(top_indices, gallery_path_list, query_path_list)


if __name__ == '__main__':
    tf.compat.v1.app.run()