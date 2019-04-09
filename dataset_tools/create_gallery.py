'''
TODO

'''

import numpy as np


def create_cmc_probe_and_gallery(data_y, camera_indices=None, seed=None):
    """Create probe and gallery images for evaluation of CMC top-k statistics.

    For every identity, this function selects one image as probe and one image
    for the gallery. Cross-view validation is performed when multiple cameras
    are given.

    Parameters
    ----------
    data_y : ndarray
        Vector of data labels.
    camera_indices : Optional[ndarray]
        Optional array of camera indices. If possible, probe and gallery images
        are selected from different cameras (i.e., cross-view validation).
        If None given, assumes all images are taken from the same camera.
    seed : Optional[int]
        The random seed used to select probe and gallery images.

    Returns
    -------
    (ndarray, ndarray)
        Returns a tuple of indices to probe and gallery images.

    """
    data_y = np.asarray(data_y)
    if camera_indices is None:
        camera_indices = np.zeros_like(data_y, dtype=np.int)
    camera_indices = np.asarray(camera_indices)

    random_generator = np.random.RandomState(seed=seed)
    unique_y = np.unique(data_y)
    probe_indices, gallery_indices = [], []
    for y in unique_y:
        mask_y = data_y == y

        unique_cameras = np.unique(camera_indices[mask_y])
        if len(unique_cameras) == 1:
            # If we have only one camera, take any two images from this device.
            c = unique_cameras[0]
            indices = np.where(np.logical_and(mask_y, camera_indices == c))[0]
            if len(indices) < 2:
                continue  # Cannot generate a pair for this identity.
            i1, i2 = random_generator.choice(indices, 2, replace=False)
        else:
            # If we have multiple cameras, take images of two (randomly chosen)
            # different devices.
            c1, c2 = random_generator.choice(unique_cameras, 2, replace=False)
            indices1 = np.where(np.logical_and(mask_y, camera_indices == c1))[0]
            indices2 = np.where(np.logical_and(mask_y, camera_indices == c2))[0]
            i1 = random_generator.choice(indices1)
            i2 = random_generator.choice(indices2)

        probe_indices.append(i1)
        gallery_indices.append(i2)

    return np.asarray(probe_indices), np.asarray(gallery_indices)