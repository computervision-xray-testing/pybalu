__all__ = ['morphoreg']
# pylint: disable=no-name-in-module
from pybalu.feature_extraction.geometric_utils import bw_perim
# pylint: enable=no-name-in-module
from skimage.morphology import binary_closing, remove_small_objects, remove_small_holes
import numpy as np
import warnings

_disk = np.ones((13, 13), dtype=np.uint8)
_disk[[0, 0, 0, 0, 1, 1, -2, -2, -1, -1, -1, -1],
      [0, 1, -2, -1, 0, -1, 0, -1, 0, 1, -2, -1]] = 0


def morphoreg(image, threshold=None):
    if threshold is None:
        R = image >= 0
    else:
        R = image > threshold

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clean = remove_small_objects(R, R.size // 100.0, connectivity=2)
        closed = binary_closing(clean, _disk)
        region = remove_small_holes(
            closed, R.size // 100.0, connectivity=2).astype(int)
        edge = bw_perim(region)

    return region, edge
