__all__ = ['hog_features', 'HOGExtractor']

import numpy as np
from scipy.signal import convolve2d
from pybalu.misc import im2row

from .hog_utils import sep_into_bins
from pybalu.base import FeatureExtractor

_hj = np.array([[1, 0, -1]])
_hi = - _hj.T


def hog_features(image, region=None, *, v_windows=0, h_windows=0, n_bins=0, normalize=False, labels=False, show=False):
    '''\
hog_features(image, region=None, *, v_windows=0, h_windows=0, n_bins=0, normalize=False, labels=False, show=False)

(TODO)

Parameters
----------
(TODO)

Returns
-------
(TODO)

See Also
--------
(TODO)

Examples
--------
(TODO)
'''
    if not isinstance(v_windows, int) or v_windows <= 0:
        raise ValueError(
            f"`v_windows` must be an integer value greater than 0")
    if not isinstance(h_windows, int) or h_windows <= 0:
        raise ValueError(
            f"`h_windows` must be an integer value greater than 0")
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError(f"`n_bins` must be an integer value greater than 0")

    if region is None:
        region = np.ones_like(image)

    Gj = convolve2d(image, _hj, mode='same')
    Gi = convolve2d(image, _hi, mode='same')

    A = np.mod(np.arctan2(Gi, Gj), np.pi)
    G = np.sqrt((Gi * Gi + Gj * Gj))
    B = np.floor_divide(A * n_bins, np.pi).astype(int)
    mags = im2row(G, v_windows, h_windows)
    bins = im2row(B, v_windows, h_windows)

    features = []

    for mag, _bins in zip(mags, bins):
        curr_feats = sep_into_bins(mag, _bins, n_bins)
        norm = (curr_feats @ curr_feats) ** .5
        curr_feats /= norm or 1
        features.append(curr_feats)

    hog_feats = np.hstack(features)

    if normalize:
        hog_feats /= hog_feats.sum()

    if labels:
        return np.array(["HOG"] * hog_feats.size), hog_feats

    return hog_feats


class HOGExtractor(FeatureExtractor):
    def __init__(self, *, v_windows=0, h_windows=0, n_bins=0, normalize=False):
        self.v_windows = v_windows
        self.h_windows = h_windows
        self.n_bins = n_bins
        self.normalize = normalize

    def transform(self, X):
        params = self.get_params()
        return np.array([hog_features(x, **params) for x in self._get_iterator(X)])

    def get_labels(self):
        return "HOG"
