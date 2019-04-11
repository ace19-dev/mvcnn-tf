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
import numpy as np

import tensorflow as tf

from retrieval import retrieval_data, matching, linear_assignment
import mvcnn

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS


# Dataset settings.
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/modelnet',
                    'Where the dataset reside.')

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

# retrieval params
flags.DEFINE_float('max_cosine_distance', 0.2,
                   'Gating threshold for cosine distance')
flags.DEFINE_string('nn_budget', None,
                    'Maximum size of the appearance descriptors gallery. '
                    'If None, no budget is enforced.')


MODELNET_RETRIEVAL_SRC_DATA_SIZE = 450
MODELNET_RETRIEVAL_RETRIEVAL_DATA_SIZE = 50


# TODO
def display_retrieval(top5):
    tf.logging.info('')


# TODO
def get_top5(cost_matrix):
    tf.logging.info('')


# TODO
def match(metric, retrieval, sources):
    def gated_metric(tracks, dets, track_indices, detection_indices):
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        cost_matrix = metric.distance(features, targets)
        # cost_matrix = linear_assignment.gate_cost_matrix(
        #     self.kf, cost_matrix, tracks, dets, track_indices,
        #     detection_indices)

        return cost_matrix

    # # Split track set into confirmed and unconfirmed tracks.
    # confirmed_tracks = [
    #     i for i, t in enumerate(self.tracks) if t.is_confirmed()]
    # unconfirmed_tracks = [
    #     i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

    # Associate targets using appearance features.
    matches_a, unmatched_tracks_a, unmatched_detections = \
        linear_assignment.matching_cascade(gated_metric,
                                           metric.matching_threshold,
                                           retrieval,
                                           sources)

    # # Associate remaining tracks together with unconfirmed tracks using IOU.
    # iou_track_candidates = unconfirmed_tracks + [
    #     k for k in unmatched_tracks_a if
    #     self.tracks[k].time_since_update == 1]
    # unmatched_tracks_a = [
    #     k for k in unmatched_tracks_a if
    #     self.tracks[k].time_since_update != 1]
    # matches_b, unmatched_tracks_b, unmatched_detections = \
    #     linear_assignment.min_cost_matching(
    #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
    #         detections, iou_track_candidates, unmatched_detections)

    matches = matches_a # + matches_b
    # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
    return matches  # , unmatched_tracks, unmatched_detections


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

        global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        # Get the number of training/validation steps per epoch
        batches = int(MODELNET_RETRIEVAL_SRC_DATA_SIZE / FLAGS.batch_size)
        if MODELNET_RETRIEVAL_SRC_DATA_SIZE % FLAGS.batch_size > 0:
            batches += 1
        batches2 = int(MODELNET_RETRIEVAL_RETRIEVAL_DATA_SIZE / FLAGS.batch_size)
        if MODELNET_RETRIEVAL_RETRIEVAL_DATA_SIZE % FLAGS.batch_size > 0:
            batches2 += 1

        source_filenames = os.path.join(FLAGS.dataset_dir, 'source.record')
        retrieval_filenames = os.path.join(FLAGS.dataset_dir, 'retrieval.record')

        source_features_list = []
        sess.run(iterator.initializer, feed_dict={tfrecord_filenames: source_filenames})
        for i in range(batches):
            batch_xs, image_paths = sess.run(next_batch)
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
            _f = sess.run([features], feed_dict={X: batch_xs})
            source_features_list.extend(_f)

        # tf.logging.info("size %d" % len(features_repo))

        # get retrieval features
        retrieval_features_list = []
        sess.run(iterator.initializer, feed_dict={tfrecord_filenames: retrieval_filenames})
        for i in range(batches2):
            batch_xs, image_paths = sess.run(next_batch)
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
            _f = sess.run([features], feed_dict={X: batch_xs})
            retrieval_features_list.extend(_f)

        # The distance metric used for measurement to retrieval.
        metric = \
            matching.NearestNeighborDistanceMetric("cosine", FLAGS.max_cosine_distance)
        # TODO:
        cost_matrix = match(metric, retrieval_features_list, source_features_list)
        top5 = get_top5(cost_matrix)

        # display top 5 image correspond to target
        display_retrieval(top5)


if __name__ == '__main__':
    tf.app.run()