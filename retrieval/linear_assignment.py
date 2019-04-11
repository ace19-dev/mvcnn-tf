from __future__ import absolute_import

import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment


# INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, retrievals, sources, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of retrievals and sources as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    retrievals : List[track.Track]
        A list of predicted retrievals at the current time step.
    sources : List[detection.Detection]
        A list of sources at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to retrievals in
        `retrievals` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        sources in `sources` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(retrievals))
    if detection_indices is None:
        detection_indices = np.arange(len(sources))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        retrievals, sources, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)

    matches, unmatched_targets, unmatched_features = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_features.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_targets.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_targets.append(track_idx)
            unmatched_features.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_targets, unmatched_features


def matching_cascade(distance_metric, max_distance, retrievals, sources,
        feature_indices=None, target_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of retrievals and sources as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    retrievals : List[track.Track]
        A list of predicted retrievals at the current time step.
    sources : List[detection.Detection]
        A list of sources at the current time step.
    feature_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to retrievals in
        `retrievals` (see description above). Defaults to all retrievals.
    target_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        sources in `sources` (see description above). Defaults to all
        sources.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if feature_indices is None:
        feature_indices = list(range(len(retrievals)))
    if target_indices is None:
        target_indices = list(range(len(sources)))

    unmatched_targets = target_indices
    matches = []
    # for level in range(cascade_depth):
    #
    # if len(unmatched_targets) == 0:  # No sources left
    #     break
    #
    # feature_indices_l = [
    #     k for k in feature_indices
    #     if retrievals[k].time_since_update == 1 + level
    # ]
    # if len(feature_indices_l) == 0:  # Nothing to match at this level
    #     continue

    matches_l, _, unmatched_targets = \
        min_cost_matching(distance_metric, max_distance, retrievals, sources,
                          feature_indices_l, unmatched_targets)

    matches += matches_l
    unmatched_features = list(set(feature_indices) - set(k for k, _ in matches))
    return matches, unmatched_features, unmatched_targets


# def gate_cost_matrix(
#         kf, cost_matrix, tracks, detections, track_indices, detection_indices,
#         gated_cost=INFTY_COST, only_position=False):
#     """Invalidate infeasible entries in cost matrix based on the state
#     distributions obtained by Kalman filtering.
#
#     Parameters
#     ----------
#     kf : The Kalman filter.
#     cost_matrix : ndarray
#         The NxM dimensional cost matrix, where N is the number of track indices
#         and M is the number of detection indices, such that entry (i, j) is the
#         association cost between `tracks[track_indices[i]]` and
#         `detections[detection_indices[j]]`.
#     tracks : List[track.Track]
#         A list of predicted tracks at the current time step.
#     detections : List[detection.Detection]
#         A list of detections at the current time step.
#     track_indices : List[int]
#         List of track indices that maps rows in `cost_matrix` to tracks in
#         `tracks` (see description above).
#     detection_indices : List[int]
#         List of detection indices that maps columns in `cost_matrix` to
#         detections in `detections` (see description above).
#     gated_cost : Optional[float]
#         Entries in the cost matrix corresponding to infeasible associations are
#         set this value. Defaults to a very large value.
#     only_position : Optional[bool]
#         If True, only the x, y position of the state distribution is considered
#         during gating. Defaults to False.
#
#     Returns
#     -------
#     ndarray
#         Returns the modified cost matrix.
#
#     """
#     gating_dim = 2 if only_position else 4
#     gating_threshold = kalman_filter.chi2inv95[gating_dim]
#     measurements = np.asarray(
#         [detections[i].to_xyah() for i in detection_indices])
#     for row, track_idx in enumerate(track_indices):
#         track = tracks[track_idx]
#         gating_distance = kf.gating_distance(
#             track.mean, track.covariance, measurements, only_position)
#         cost_matrix[row, gating_distance > gating_threshold] = gated_cost
#     return cost_matrix
